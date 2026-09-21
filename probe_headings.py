"""What do 10-Q Item 1 / Item 2 headings actually say? Read-only."""
import csv, re
from collections import Counter
from tools.config import load_config
from tools.edgar import from_stored_path
from tools.htmltext import extract_text, inline_tags_from_config

cfg = load_config()
inline = inline_tags_from_config(cfg)
man = list(csv.DictReader(open("inputs/manifest.csv", encoding="utf-8")))
led = {(r["accession"], r["document"]): r
       for r in csv.DictReader(open("temp/filings/fetched.csv", encoding="utf-8"))}

# Every "Item <n>" occurrence and the 60 chars that follow it.
PAT = re.compile(r"item\s*(\d+a?)\s*[-–—.:]?\s*(.{0,58})", re.I)
by_item = {}
toc_gap = Counter()
for row in [r for r in man if r["form"] == "10-Q" and r["doc_type"] == "primary"]:
    e = led[(row["accession"], row["document"])]
    text = extract_text(from_stored_path(e["path"]).read_bytes(), inline_tags=inline)
    for m in PAT.finditer(text):
        num = m.group(1).upper()
        tail = re.sub(r"\s+", " ", m.group(2)).strip()
        # Normalize digits so "33 SIGNATURES 35" style TOC tails collapse.
        by_item.setdefault(num, Counter())[tail[:52]] += 1

for num in sorted(by_item, key=lambda n: (len(n), n)):
    print(f"\n--- Item {num}: {sum(by_item[num].values())} occurrences, "
          f"{len(by_item[num])} distinct continuations")
    for tail, n in by_item[num].most_common(6):
        print(f"    {n:>4}  {tail!r}")
