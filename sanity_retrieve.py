"""Sanity check on the built index. Structural verification only, per Rule 6:
it confirms rows, metadata and source text line up -- it does not interpret what
any filing says.
"""
import json
from pathlib import Path

import faiss
import numpy as np

from tools.config import load_config
from tools.edgar import from_stored_path
from tools.embedders import get_embedder
from tools.htmltext import extract_text, inline_tags_from_config

cfg = load_config()
d = Path("outputs/index/main")
idx = faiss.read_index(str(d / "index.faiss"))
meta = [json.loads(l) for l in (d / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
side = json.loads((d / "sidecar.json").read_text(encoding="utf-8"))
emb = get_embedder(cfg)
top_k = int(cfg.get("retrieve.top_k"))

print("=== 1. round-trip: a chunk's own text must retrieve its own row ===")
ok = 0
for row in (0, 1234, 4321, 8689):
    v = emb.embed_documents([meta[row]["text"]])
    _, ids = idx.search(np.ascontiguousarray(v, dtype="float32"), 1)
    hit = int(ids[0][0])
    ok += hit == row
    print(f"  row {row:>5} -> returned {hit:>5}  {'OK' if hit == row else 'MISALIGNED'}")
print(f"  {ok}/4 aligned")

print()
print(f"=== 2. query search, top_k={top_k} (showing 5) ===")
q = "How did quarterly net sales and operating income change?"
qv = emb.embed_query(q).reshape(1, -1)
scores, ids = idx.search(np.ascontiguousarray(qv, dtype="float32"), top_k)
print(f"  query: {q!r}")
print(f"  {'score':>7}  {'ticker':<6} {'form':<6} {'section':<34} filed")
for s, i in zip(scores[0][:5], ids[0][:5]):
    m = meta[int(i)]
    print(f"  {s:>7.4f}  {m['ticker']:<6} {m['form']:<6} {str(m['section'])[:34]:<34} {m['filing_date']}")
print(f"  returned {len(ids[0])} hits; scores in [-1,1] as cosine: "
      f"{bool(scores[0].max() <= 1.0001 and scores[0].min() >= -1.0001)}")

print()
print("=== 3. provenance: char_span must locate the chunk in its source file ===")
led = {}
import csv
with open("temp/filings/fetched.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        led[(r["accession"], r["document"])] = r
inline = inline_tags_from_config(cfg)
checked = 0
for row in (0, 2500, 6000, 8689):
    m = meta[row]
    entry = led[(m["accession"], m["document"])]
    text = extract_text(from_stored_path(entry["path"]).read_bytes(), inline_tags=inline)
    start, end = m["char_span"]
    located = text[start:end].strip() == m["text"]
    checked += located
    print(f"  row {row:>5}  {m['ticker']:<5} {m['document'][:36]:<36} span {start}..{end}  "
          f"{'matches source' if located else 'DOES NOT MATCH'}")
print(f"  {checked}/4 chunks located verbatim in their source document")

print()
print("=== 4. sidecar identifies what produced the index ===")
for k, v in side["embedder"].items():
    print(f"  {k:16} {v}")
