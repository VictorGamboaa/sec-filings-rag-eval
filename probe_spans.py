"""Where do the 10-Q section boundaries actually land? Read-only."""
import csv, re
from pathlib import Path
from tools.config import load_config
from tools.edgar import from_stored_path
from tools.htmltext import extract_text, inline_tags_from_config
from tools.parse_chunk import _anchors_for, split_sections

cfg = load_config()
inline = inline_tags_from_config(cfg)
man = list(csv.DictReader(open("inputs/manifest.csv", encoding="utf-8")))
led = {(r["accession"], r["document"]): r
       for r in csv.DictReader(open("temp/filings/fetched.csv", encoding="utf-8"))}

tenqs = [r for r in man if r["form"] == "10-Q" and r["doc_type"] == "primary"]
for row in tenqs[:3]:
    e = led[(row["accession"], row["document"])]
    text = extract_text(from_stored_path(e["path"]).read_bytes(), inline_tags=inline)
    spans = split_sections(
        text,
        _anchors_for(cfg, "10-Q", "primary"),
        min_section_chars=int(cfg.get("parse.min_section_chars")),
        preamble_label=cfg.get("parse.preamble_label"),
    )
    print(f"\n{'='*82}\n{row['ticker']} {row['filing_date']}  text={len(text):,} chars")
    for label, s, en in spans:
        head = text[s:s+70].replace("\n", " ")
        print(f"  {str(label):<36} {s:>8,}..{en:>8,} ({en-s:>8,})  {head!r}")
