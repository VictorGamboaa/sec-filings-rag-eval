# Dense retrieval on SEC filings: a measured negative result

A RAG pipeline over 363 SEC documents, evaluated against a hand-verified answer key. The short version: dense vector retrieval alone finds about 43% of the answers, and neither a 3x larger embedding model nor a commercial finance-domain model moves that number. Adding lexical retrieval alongside it raises recall@20 by half. No configuration tested produces a usable "I don't know" signal.

This repository exists because the pipeline did not work well and the reasons turned out to be more interesting than a working pipeline would have been.

## What was built

A four-stage pipeline over 10-Q, 8-K, and 8-K exhibit filings from five companies (BCPC, AGNC, AIG, ACI, HON), filed 2024 onward.

```
discover  ->  fetch  ->  parse + chunk  ->  embed + index
```

Each stage writes its output to disk and skips work already done, keyed on `(accession, document)` plus a fingerprint of the config that produced it. Changing the chunk size costs one stage re-run, not a full re-fetch from EDGAR.

| | |
|---|---|
| Corpus | 363 documents (224 filings, 139 exhibits), 5 issuers |
| Chunking | 450 tokens, 90 overlap, snapped to sentence boundaries |
| Chunks | 8,690 |
| Embeddings | BAAI/bge-small-en-v1.5 (33M params, 384 dim), CPU |
| Index | FAISS `IndexFlatIP` over L2-normalized vectors (exact cosine, no approximation) |
| Retrieval | top-k 20 |

Every chunk carries full provenance: CIK, ticker, accession, form type, filing date, period of report, document filename, exhibit label, section anchor, and chunk ordinal. Retrieval results are traceable to a specific line in a specific filing.

## How it was evaluated

A 26-entry answer key, written by hand and verified verbatim against the fetched documents before any scoring ran.

| type | n | what it tests |
|---|---|---|
| `single_fact` | 14 | Can retrieval find one specific disclosure |
| `period_over_period` | 5 | Does it surface both filings a comparison needs |
| `negative_control` | 5 | Does it signal when the answer is not in the corpus |
| `cross_company` | 1 | Does it surface the same disclosure across two issuers |
| `dedup_diagnostic` | 1 | Does duplicated boilerplate crowd out real content |

The negative controls are the part most evaluations skip and they turned out to matter most. Four of the five are designed to be plausible but unanswerable: a company outside the universe, a period outside the window, a disclosure type that lives in proxy statements rather than the forms ingested, and a segment that does not exist (Balchem has three segments, none of them semiconductor).

A verification tool (`tools/verify_key.py`) checks that every ground-truth quote appears verbatim in the indexed chunks before evaluation runs. This caught three paraphrased quotes that would have silently scored as retrieval failures. It runs against each index independently, because a quote that fits a 450-token chunk may not fit a 200-token one.

## Results

Four experiments. Three hypotheses falsified, one confirmed.

### Everything measured

| metric | 450 / bge-small | 200 / bge-small | 450 / bge-base | voyage-finance-2 | BM25 | bge hybrid | voyage hybrid |
|---|---|---|---|---|---|---|---|
| single_fact recall@5 | 0.2857 | 0.1429 | 0.1429 | 0.1429 | 0.1429 | 0.2143 | **0.3571** |
| single_fact recall@20 | 0.4286 | 0.2857 | 0.3571 | 0.4286 | 0.5714 | **0.6429** | 0.5714 |
| top-1 correct | 0.0714 | 0.0000 | 0.0714 | 0.0000 | 0.0000 | **0.1429** | 0.0000 |
| citation correct | 0.2143 | 0.1429 | 0.1429 | 0.1429 | 0.0714 | **0.2857** | 0.2143 |
| section accuracy | 0.2857 | 0.7143 | 0.5714 | 0.1429 | 0.7143 | **0.7143** | 0.4286 |
| cross_company recall@5 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| cross_company recall@20 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| period coverage recall@20 | 0.4000 | 0.8000 | 0.4000 | 0.6000 | 0.4000 | 0.6000 | 0.6000 |

Section accuracy is over n=7 (BCPC and ACI, the two issuers with reliable section labelling) and cross_company is a single question. Treat both as directional. BM25 and RRF scores are not comparable across queries, so the score-based metrics below are reported only for the cosine configurations.

### Smaller chunks made it worse

Halving the chunk size from 450 to 200 tokens cut single-fact recall@5 in half and dropped top-1 correctness to zero. The likely mechanism: at 450 tokens a chunk holds both the entity and its attribute ("Carrier Global Access Solutions" and "$4,913 million" in the same window). At 200 they land in different chunks, and a question asking for one in terms of the other can match neither.

Coverage metrics improved at 200, but that improvement is partly mechanical. There are 2.25x more chunks per filing, which raises the odds that some chunk from the right filing lands in top-20 regardless of whether it is relevant. Only section accuracy is worth taking at face value.

### A 3x larger model made it worse

Swapping bge-small (33M params, 384 dim) for bge-base (110M, 768 dim) at the same chunk size. Same family, so the asymmetric query prefix convention is unchanged; same tokenizer, verified entry by entry across all 30,522 vocab terms, so the existing chunks were re-embedded without re-parsing.

Single-fact recall@5 halved. Citation correctness halved. Top-1 was flat at 1 of 14.

The separation metrics appear to improve (median separation 0.0400 to 0.0618), but that is an artifact of the whole score distribution compressing downward, not of answers becoming more distinguishable from non-answers. The metric that matters got worse: answerable questions outscored by the best negative control went from 10 of 20 to 13 of 20.

### A commercial finance-domain model did not help either

`voyage-finance-2` is built for exactly this document type: 1024 dimensions, domain-tuned, and it uses `input_type` rather than a query prefix, so no prompt convention had to be guessed at. Same 8,690 chunks, same fingerprint, re-embedded only.

Recall@20 came out at 0.4286, identical to bge-small. Recall@5, top-1, citation correctness and section accuracy were all lower. It recovered one of the seven questions invisible to both bge configurations, and that one at rank 17.

The chunks are a caveat here and it is worth stating plainly rather than leaving for a reader to raise. 450 tokens was chosen to fit bge's 512-token ceiling; voyage accepts 32,000, and under its own tokenizer these chunks run about 2x longer than bge recorded, with a maximum of 871. So voyage is being handed passages far smaller than it could take. Whether it would do better on larger chunks is untested, and is the obvious next experiment.

What the result does establish is narrower but still useful: at a chunk size that suits a general-purpose model, domain tuning does not recover the questions dense retrieval misses.


### Dense and lexical retrieval fail on different questions

This is the main result.

Of the 14 single-fact questions, dense retrieval (bge-small, 450) finds 6 and BM25 finds 6. **The overlap is 2.** The union is 10, and reciprocal rank fusion recovers 9 of those 10.

The pattern holds across dense models. Rank of the target chunk, by configuration:

| id | bge-small | bge-base | voyage | BM25 | bge hybrid | voyage hybrid |
|---|---|---|---|---|---|---|
| q01 | 4 | 5 | 2 | 8 | 4 | 4 |
| q02 | 4 | 12 | 19 | -- | 16 | -- |
| q03 | -- | -- | -- | **3** | -- | 7 |
| q04 | -- | -- | -- | -- | -- | -- |
| q05 | -- | 13 | 12 | **7** | 11 | 5 |
| q06 | -- | -- | -- | -- | -- | -- |
| q07 | -- | -- | -- | -- | -- | -- |
| q08 | 11 | -- | -- | **3** | 6 | 11 |
| q09 | -- | -- | -- | **7** | 9 | 6 |
| q10 | 7 | -- | -- | 12 | 6 | -- |
| q11 | 4 | 15 | 3 | 10 | 1 | 3 |
| q12 | -- | -- | -- | -- | -- | -- |
| q27 | -- | -- | 17 | **8** | 8 | 2 |
| q28 | 1 | 1 | 6 | -- | 1 | 4 |

`--` means the target chunk was outside the top 20.

**Six of the fourteen questions (q03, q04, q06, q07, q09, q12) are invisible to every dense configuration tested.** That is three models spanning 33M parameters to a commercial domain-tuned system, two chunk sizes, and 384 to 1024 dimensions. BM25 finds two of the six immediately. Four (q04, q06, q07, q12) are found by nothing.

The questions BM25 recovers ask about named entities and specific figures: a June 2025 acquisition, a joint venture partner, a CEO quote from an earnings release. That is exactly the signal a dense embedding averages away and exactly the signal BM25 keys on. It is not a model quality problem, and three models agreeing is the evidence for that.

Fusion has a cost worth noting. Under equal-weight RRF a confident minority retriever can be outvoted: q03 ranks 3rd under BM25 and drops out of the top 20 entirely in bge hybrid. That is also why bge hybrid's recall@5 fell slightly even as recall@20 rose, since fusion redistributes results into the 5 to 20 band. voyage hybrid does not have this problem to the same degree and posts the best recall@5 of any configuration at 0.3571.

### The confidence signal never worked

Retrieval always returns something. On a corpus of SEC filings, it returns something that looks plausible.

| | bge-small | bge-base | voyage |
|---|---|---|---|
| answerable median | 0.7771 | 0.7262 | 0.5699 |
| negative control max | 0.7786 | 0.7329 | 0.5508 |
| median separation | 0.0400 | 0.0618 | **0.1105** |
| distributions overlap | yes | yes | yes |
| answerable outscored by best NC | 10/20 | 13/20 | **9/20** |

voyage produced the widest separation and the fewest answerable questions losing to an unanswerable one. That is a real improvement rather than the compression artifact bge-base showed. It is also not enough: the distributions still overlap, and nine of twenty answerable questions still score below the best negative control. A threshold that rejected the worst unanswerable question would reject nearly half the answerable ones with it.

Hybrid retrieval does nothing here either. The highest-scoring negative control under bge RRF (0.0311) beats 6 of the 14 answerable questions.

A concrete illustration, from the pipeline diagnostics rather than the evaluation: querying with the exact text of a chunk returned that chunk at rank 1 in only 7 of 8 trials. The miss was a chunk inside a 31-way identical boilerplate cluster, where a near-variant of the same text beat the true chunk by 0.0024. If verbatim text cannot reliably beat boilerplate, a natural-language question has no chance.

These are two separate defects and the second is worse. A system that surfaces the right passage 64% of the time is workable if it knows which 64%. A system that cannot tell a correct retrieval from confidently-retrieved boilerplate is not deployable at any recall.

## What this says about growing the corpus

A common assumption is that retrieval precision improves as a corpus grows, since more documents mean more context and metadata filters keep growth from diluting relevance.

That did not hold here. Growth in a filing corpus is mostly duplicated boilerplate. One block of Rule 13e-4(c) language appears across 34 accessions in 3 distinct variants. Honeywell's Section 12(b) securities table prints identically on the cover of every 10-Q and 8-K it files. Every one of the five negative controls returned an EX-99 exhibit as its top hit, four of them press releases, regardless of what was asked.

More filings means more of this, and dense similarity has no way to tell one instance from another.

## Design choices

Some notable decisions and what drove them.

**Local embeddings as the default, with a hosted model tested against them.** The bge configurations need no vendor account and no billing credential, so anyone cloning this repository can reproduce them. `voyage-finance-2` was run as a control rather than as the default, to test whether domain tuning is what the general-purpose models were missing. It was not: see the results above. Reproducing the voyage arm requires an API key, and the cost appendix below has the measured figures.

**Exact search instead of approximate.** FAISS `IndexFlatIP` is exhaustive. Approximate indexes (HNSW and similar) are faster and are the common default, but they introduce a recall error that would be indistinguishable from the retrieval failures being measured.

**EDGAR HTML instead of PDFs.** PDF parsing is the largest CPU cost in a filings pipeline and EDGAR does not serve PDFs natively. Fetching HTML directly removes the bottleneck and the parsing error class that comes with it.

**Exhibits as first-class documents.** 114 of 184 8-Ks referenced an EX-99 exhibit that the initial manifest did not fetch, and for Item 2.02 earnings releases it was 60 of 60. About 61% of 8-K text volume was a pointer to a document that was not being ingested. The manifest was restructured to one row per document rather than per filing, adding 139 exhibits.

**Chunk size 450, not 800.** bge-small has a hard 512-token ceiling, verified against the model's own `sentence_bert_config.json` and `config.json`. Two of those tokens are special tokens. Anything longer is silently truncated at embed time with no error and no warning, so the index reports a full chunk count while the back half of every long chunk is missing from its vector. 450 leaves headroom, and the embed tool refuses an over-length chunk rather than truncating it.

The pipeline architecture follows a published RAG walkthrough for financial documents. The embedding model, vector store, chunk size, and input format were all changed for the reasons above, so these results describe this implementation rather than that one.

## Known issues

- `n_tokens` in the chunk metadata undercounts by 1 to 2 tokens on 148 of 8,690 chunks. The windower drops zero-width offset tokens when counting. It does not threaten the 512 ceiling at the sizes used (measured max 452), but it is an undercount, which is the direction that would matter if anything downstream used it to assert a chunk fits.
- q26 (the dedup diagnostic) has a 1,289-character ground-truth quote and is unscoreable at chunk size 200 by construction. A shorter span was tested and rejected: every candidate under 200 characters resolves to a wider accession set, because the distinguishing text sits outside any short window.
- Section labelling is unreliable for AGNC (0 of 8 filings labelled) and was excluded from section-accuracy scoring.
- On `main_voyage`, re-embedding a stored chunk reproduces its vector to cosine 0.99996 but not to `atol=1e-5`, which is the tolerance the local models meet. The API is bit-identical on two successive calls within a session and is unaffected by batch position, and two identical searches return the same top-20 in the same order, so retrieval is not affected. The cause has not been established and the assertion was deliberately not loosened, since the local models do reproduce bit-for-bit and that check should keep catching it if they stop.
- voyage dense is the only configuration to score 0.0000 on cross_company recall@20, where every other configuration including voyage hybrid scores 1.0000. It also posts the lowest section accuracy of the six at 0.1429. Both are single-question or n=7 metrics and are recorded rather than explained.
- Under BM25, the number of chunks scoring above zero varied from 3,974 to 8,650 across the negative controls, with the most implausible question (a nonexistent segment) scoring the fewest. Lexical sparsity may be a better unanswerability signal than score. This was not tested against the answerable questions and is an open question, not a finding.

## Repository layout

```
inputs/
  config.yaml          all tunables, no constants in code
  universe.csv         ticker, cik, company name
  manifest.csv         one row per document
  answer_key.yaml      26 verified entries
tools/
  discover.py          ticker list -> filing manifest
  fetch.py             rate-limited EDGAR retrieval
  parse_chunk.py       HTML -> text -> chunks with provenance
  embed_index.py       batched embedding, crash-consistent writes
  retrieve.py          dense, BM25, and fused retrieval
  evaluate.py          scores against the answer key
  verify_key.py        verifies every quote before scoring
  audit_corpus.py      corpus completeness checks
outputs/index/
  main                 450 tokens, bge-small
  main_200             200 tokens, bge-small
  main_base            450 tokens, bge-base
  main_bm25            450 tokens, lexical
  main_voyage          450 tokens, voyage-finance-2
```

Each index carries a sidecar recording its embedder fingerprint, chunk fingerprint, and parameters. The evaluation binds to that fingerprint, so results cannot be silently compared across incompatible builds.

## Reproducing

```bash
uv sync
export EDGAR_USER_AGENT="Your Name your@email.com"
python tools/discover.py
python tools/fetch.py
python tools/parse_chunk.py
python tools/embed_index.py --index main
python tools/verify_key.py --index main
python tools/evaluate.py --index main
```

Fetching respects SEC's fair-access rate limit. A full build from empty takes a few hours on a laptop, most of it in fetch and embed.

## What would come next

Not run, and listed so the gaps are explicit rather than implied:

- voyage-finance-2 on larger chunks. These chunks were sized for bge's 512-token ceiling and voyage accepts 32,000, so the domain model has not been tested in a configuration that suits it. This is the most direct open question.
- Weighted fusion instead of equal-weight RRF, which would address the q03 case
- A cross-encoder reranker over the fused candidate set, which can reorder what retrieval surfaces but cannot rescue a target that never enters the top 20
- Deduplication before indexing, and a scaling curve at 25/50/100% of the corpus
- Whether lexical sparsity (the count of chunks scoring above zero under BM25) separates answerable from unanswerable questions, which the negative controls hint at but do not establish

## Cost

Measured, not estimated from a rate card. The bge and BM25 configurations cost nothing beyond CPU time.

| | |
|---|---|
| voyage-finance-2, full build | 4,443,914 tokens, 8,690 chunks |
| average per chunk | 511.4 tokens |
| API requests | 68, batch size 128 |
| billed | $0.00 |
| list rate | $0.12 per million, so $0.53 if the free allowance had been exhausted |
| free allowance consumed | 8.9% of 50M, leaving roughly 11 further rebuilds |

Extrapolating linearly, a universe 100x this one would embed at roughly 444M tokens, about $53 at list rate for a full rebuild, less for incremental updates. Embedding cost is not what makes this approach hard.

## License

MIT for the code. The SEC filings are public domain. The answer key is included and contains verbatim quotes from those filings.
