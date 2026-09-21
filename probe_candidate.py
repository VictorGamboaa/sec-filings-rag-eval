"""Measure a candidate Item 1 anchor against the real 10-Q corpus. Read-only."""
import csv
from pathlib import Path
from tools.config import load_config
from tools.edgar import from_stored_path
from tools.htmltext import extract_text, inline_tags_from_config
from tools.parse_chunk import _anchors_for, split_sections

cfg = load_config()
inline = inline_tags_from_config(cfg)
minc = int(cfg.get("parse.min_section_chars"))
pre = cfg.get("parse.preamble_label")
man = list(csv.DictReader(open("inputs/manifest.csv", encoding="utf-8")))
led = {(r["accession"], r["document"]): r
       for r in csv.DictReader(open("temp/filings/fetched.csv", encoding="utf-8"))}

CURRENT = r"item\s*1\.?\s*[-–—:]?\s*financial\s+statements"
# Allow the words filers actually put between the number and the title
# ("Condensed Consolidated", "Unaudited Condensed"), observed in the corpus.
# The running page header "Item 1 | Notes to ... Financial Statements" is
# excluded by construction: "|" is not a word character, so the optional word
# run cannot cross it.
CANDIDATE = r"item\s*1\.?\s*[-–—:]?\s*(?:[A-Za-z()]+\s+){0,4}financial\s+statements"

texts = []
for row in [r for r in man if r["form"] == "10-Q" and r["doc_type"] == "primary"]:
    e = led[(row["accession"], row["document"])]
    texts.append((row["ticker"],
                  extract_text(from_stored_path(e["path"]).read_bytes(), inline_tags=inline)))

for name, pattern in (("CURRENT", CURRENT), ("CANDIDATE", CANDIDATE)):
    anchors = []
    for a in _anchors_for(cfg, "10-Q", "primary"):
        anchors.append({**a, "pattern": pattern} if a["label"] == "part1_item1_financial_statements" else a)
    docs_with_fin = 0
    unsec = 0
    total = 0
    fin_chars = 0
    per_ticker = {}
    for ticker, text in texts:
        spans = split_sections(text, anchors, min_section_chars=minc, preamble_label=pre)
        labels = {s[0] for s in spans}
        hit = "part1_item1_financial_statements" in labels
        docs_with_fin += hit
        per_ticker.setdefault(ticker, [0, 0])
        per_ticker[ticker][0] += hit
        per_ticker[ticker][1] += 1
        for label, s, en in spans:
            total += en - s
            if label is None:
                unsec += en - s
            elif label == "part1_item1_financial_statements":
                fin_chars += en - s
    print(f"\n=== {name} ===")
    print(f"  docs with a financial-statements section : {docs_with_fin}/{len(texts)}")
    print(f"  chars in that section                    : {fin_chars:,} ({100*fin_chars/total:.1f}%)")
    print(f"  unsectioned                              : {unsec:,} ({100*unsec/total:.1f}%)")
    print(f"  per ticker: " + ", ".join(f"{t}={v[0]}/{v[1]}" for t, v in sorted(per_ticker.items())))
