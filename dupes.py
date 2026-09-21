"""Exact-duplicate chunk census. Counts only -- no interpretation (Rule 6)."""
import json
from collections import Counter
from pathlib import Path

meta = [json.loads(l) for l in
        Path("outputs/index/main/metadata.jsonl").read_text(encoding="utf-8").splitlines()]
c = Counter(m["text"] for m in meta)
dupe_groups = {t: n for t, n in c.items() if n > 1}
dupe_rows = sum(n for n in dupe_groups.values())
print(f"total chunks          : {len(meta):,}")
print(f"distinct texts        : {len(c):,}")
print(f"chunks in a dup group : {dupe_rows:,} ({100*dupe_rows/len(meta):.1f}%)")
print(f"redundant chunks      : {dupe_rows - len(dupe_groups):,} "
      f"({100*(dupe_rows-len(dupe_groups))/len(meta):.1f}% of the index)")
print()
by_group = Counter()
for m in meta:
    if c[m["text"]] > 1:
        by_group[f"{m['form']}/{m['doc_type']}"] += 1
print("duplicated chunks by form/doc_type:", dict(by_group))
print()
print("largest duplicate groups:")
for text, n in sorted(dupe_groups.items(), key=lambda kv: -kv[1])[:5]:
    rows = [m for m in meta if m["text"] == text]
    tickers = sorted({r["ticker"] for r in rows})
    print(f"  x{n:<4} {len(text):>5} chars  tickers={tickers}  {text[:70]!r}")
