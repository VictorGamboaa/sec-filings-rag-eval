"""Is row 4321 a misalignment, or two chunks with identical text?"""
import json
from pathlib import Path

import faiss
import numpy as np

d = Path("outputs/index/main")
idx = faiss.read_index(str(d / "index.faiss"))
meta = [json.loads(l) for l in (d / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]

a, b = meta[4321], meta[3064]
print("row 4321:", a["chunk_id"], "|", a["ticker"], a["filing_date"], a["section"])
print("row 3064:", b["chunk_id"], "|", b["ticker"], b["filing_date"], b["section"])
print()
print("texts identical:", a["text"] == b["text"])
print("len 4321:", len(a["text"]), " len 3064:", len(b["text"]))
print()
print("4321 text[:200]:", repr(a["text"][:200]))
print("3064 text[:200]:", repr(b["text"][:200]))
print()
# Vectors stored at those rows
v4321 = idx.reconstruct(4321)
v3064 = idx.reconstruct(3064)
print("stored vectors identical:", bool(np.allclose(v4321, v3064, atol=1e-6)))
print("cosine(stored 4321, stored 3064):", float(v4321 @ v3064))
print()
# How many chunks in the whole corpus share this exact text?
same = [i for i, m in enumerate(meta) if m["text"] == a["text"]]
print(f"rows sharing this exact text: {len(same)} -> {same[:10]}")
print()
# The decisive test: does the vector STORED at row 4321 correspond to the text
# metadata claims is at row 4321?
from tools.config import load_config
from tools.embedders import get_embedder
emb = get_embedder(load_config())
fresh = emb.embed_documents([a["text"]])[0]
print("cosine(fresh embed of meta[4321].text, stored vector at row 4321):",
      round(float(fresh @ v4321), 6))
print("  -> 1.0 means the row holds the vector its metadata describes")
