"""HTML -> text extraction. Shared by the corpus audit and, later, stage 3.

One extractor, used by everything that needs text out of a filing, so that a
correctness property cannot hold in one caller and silently fail in another.

THE SEPARATOR REQUIREMENT
=========================

Text nodes MUST NOT be concatenated without a separator at element boundaries.
This is a correctness requirement, not a formatting preference.

The markup EDGAR actually produces looks like this::

    <span ...>Item&#160;5.02</span></td><td ...><span ...>Departure of Directors</span>

Joined without a separator that becomes ``Item 5.02Departure of Directors``.
Every configured section anchor in ``parse.section_anchors`` has the same shape --
an item number, then whitespace, then heading words. For 10-Q::

    item\\s*2\\.?\\s*[-...:]?\\s*management.s\\s+discussion

The ``\\s+`` before the heading cannot match a weld. So a missing separator does
not produce a visible error: every 10-Q anchor misses, section-aware chunking
falls back to ``_default`` across the entire corpus, and the only symptom is
retrieval that quietly performs worse than it should. For an evaluation harness
whose purpose is measuring retrieval quality, that is the worst available failure
mode -- the instrument would be miscalibrated with no indication.

The hazard is identical for 8-K item codes and 10-Q section headings. It was
found in 8-Ks only because those were audited first.

WHY NOT A BLANKET SEPARATOR
===========================

Separating at *every* element boundary repairs the weld but breaks the opposite
case: ``<b>Fin</b>ancial`` becomes ``Fin ancial``. So a small set of purely
typographic inline tags -- the ones that appear *within* a word -- are joined
without a separator. That set is configuration
(``parse.inline_tags_no_separator``), because which tags a filer uses mid-word is
a property of the documents, not of this code.

``span`` is deliberately NOT in that set: EDGAR's generated HTML uses it for
phrase-level layout, which is exactly where the weld occurs.

HIDDEN CONTENT
==============

Inline XBRL carries a large hidden fact block (``ix:header`` / ``ix:hidden``),
and filers use ``display:none`` for the same purpose. Counting it as body text
would make a document that is a bare pointer to an exhibit look substantive,
which would defeat the audit that this module exists to serve.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:  # pragma: no cover
    from tools.config import Config

__all__ = [
    "DEFAULT_INLINE_TAGS",
    "DROP_TAGS",
    "extract_text",
    "inline_tags_from_config",
]

#: Content inside these never reaches the output.
DROP_TAGS: frozenset[str] = frozenset(
    {
        "script",
        "style",
        "head",
        "title",
        # Inline XBRL machinery: facts, not prose.
        "ix:header",
        "ix:hidden",
        "ix:references",
        "ix:resources",
    }
)

#: Fallback when config does not supply ``parse.inline_tags_no_separator``.
DEFAULT_INLINE_TAGS: frozenset[str] = frozenset(
    {"b", "i", "u", "em", "strong", "sup", "sub", "font", "small", "big"}
)

#: Void elements never carry an end tag; a stack that waits for one desynchronizes.
_VOID_TAGS: frozenset[str] = frozenset(
    {
        "area", "base", "basefont", "br", "col", "embed", "hr", "img", "input",
        "isindex", "link", "meta", "param", "source", "track", "wbr",
    }
)

_HIDDEN_RE = re.compile(r"display\s*:\s*none", re.I)
_WS_RE = re.compile(r"\s+")


def inline_tags_from_config(config: Config) -> frozenset[str]:
    """Read ``parse.inline_tags_no_separator``, falling back to the default set."""
    configured: Iterable[str] | None = config.get("parse.inline_tags_no_separator", None)
    if not configured:
        return DEFAULT_INLINE_TAGS
    return frozenset(str(tag).strip().lower() for tag in configured if str(tag).strip())


class _TextExtractor(HTMLParser):
    """Collects visible text, separating at element boundaries.

    The separator is emitted as a sentinel rather than a literal space so that
    whitespace collapsing at the end cannot tell an inserted boundary apart from
    one that was in the source -- both become a single space, which is what a
    downstream regex expects.
    """

    def __init__(self, inline_tags: frozenset[str]) -> None:
        super().__init__(convert_charrefs=True)
        self.inline_tags = inline_tags
        self.parts: list[str] = []
        self._drop_depth = 0
        self._hidden_depth = 0
        # (tag, kind) where kind is "drop" | "hide" | "keep"
        self._stack: list[tuple[str, str]] = []

    # -- boundary handling -------------------------------------------------

    def _boundary(self, tag: str) -> None:
        """Emit a separator unless this tag is one that appears within a word."""
        if tag not in self.inline_tags:
            self.parts.append(" ")

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        self._boundary(tag)
        if tag in _VOID_TAGS:
            # No end tag will arrive; nothing to push. Its content-less nature
            # means drop/hide state is irrelevant.
            return
        style = ""
        for key, value in attrs:
            if key and key.lower() == "style" and value:
                style = value
                break
        if tag in DROP_TAGS:
            self._drop_depth += 1
            self._stack.append((tag, "drop"))
        elif _HIDDEN_RE.search(style):
            self._hidden_depth += 1
            self._stack.append((tag, "hide"))
        else:
            self._stack.append((tag, "keep"))

    def handle_startendtag(self, tag, attrs):
        # <td ... /> and friends: a boundary, but it opens nothing.
        self._boundary(tag.lower())

    def handle_endtag(self, tag):
        tag = tag.lower()
        self._boundary(tag)
        if tag in _VOID_TAGS:
            return
        # Filings contain unclosed tags. Unwind to the matching open tag rather
        # than assuming balance; anything left above it was never closed.
        for i in range(len(self._stack) - 1, -1, -1):
            open_tag, kind = self._stack[i]
            if open_tag == tag:
                for _, unwound in self._stack[i:]:
                    if unwound == "drop":
                        self._drop_depth = max(0, self._drop_depth - 1)
                    elif unwound == "hide":
                        self._hidden_depth = max(0, self._hidden_depth - 1)
                del self._stack[i:]
                return
        # No matching open tag: a stray close. Ignore it.

    # -- text --------------------------------------------------------------

    def handle_data(self, data):
        if self._drop_depth == 0 and self._hidden_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        # Non-breaking spaces are whitespace for our purposes: filers write
        # "Item&#160;5.02", and an anchor pattern using \s must match it.
        joined = "".join(self.parts).replace("\xa0", " ")
        return _WS_RE.sub(" ", joined).strip()


def extract_text(
    source: str | bytes,
    *,
    inline_tags: frozenset[str] | None = None,
) -> str:
    """Extract visible body text from an HTML document.

    Guarantees, each covered by a test in ``tests/test_htmltext.py``:

    * adjacent text nodes in separate elements never weld
      (``<span>Item</span><span>2.02</span>`` -> ``"Item 2.02"``)
    * typographic inline tags do not split a word
      (``<b>Fin</b>ancial`` -> ``"Financial"``)
    * script, style and inline-XBRL hidden facts do not appear
    * ``&#160;`` and friends become ordinary spaces, so ``\\s`` matches them
    """
    if isinstance(source, bytes):
        source = source.decode("utf-8", "replace")
    parser = _TextExtractor(
        inline_tags if inline_tags is not None else DEFAULT_INLINE_TAGS
    )
    try:
        parser.feed(source)
        parser.close()
    except Exception:
        # A malformed document yields the text recovered so far. Raising would
        # discard a filing over markup we do not control; the caller counts and
        # logs a short result instead.
        pass
    return parser.text()
