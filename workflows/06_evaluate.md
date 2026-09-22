# Workflow — Evaluation harness

## Objective

Score retrieval against `inputs/answer_key.yaml` and produce the measurement the
harness exists for.

## Inputs

| Input | Source |
|---|---|
| `inputs/answer_key.yaml` | verified by `tools/verify_key.py` |
| `outputs/index/<name>/` | `workflows/04_embed_index.md` |
| `outputs/eval/key_verification_latest.json` | **written by `verify_key` — required** |
| `evaluate.*`, `retrieve.top_k` | `inputs/config.yaml` |

## Tool

```
python -m tools.verify_key      # REQUIRED FIRST -- writes the binding record
python -m tools.evaluate
python -m tools.evaluate --only q13,q19
```

Runs in WSL. Local files only; `requests: 0`.

## Run verify_key first — this is enforced, not advised

A key entry resolves to **chunk ids**, and those are valid only for the index
that produced them. Rebuilding the index — deduplicating, re-chunking, changing
the embedder — changes or removes them. An evaluator that did not notice would
resolve fewer targets, score them as misses, and report a confident number
describing nothing.

So `verify_key` records an `index_fingerprint` (a hash over the sidecar's
embedder, chunk fingerprint, dim, metric and **count**), and `evaluate`
**refuses to run** on a mismatch:

```
ERROR: index fingerprint mismatch.
  key was verified against : 7614b05c13658707
  index on disk now is     : <other>
Re-run `python -m tools.verify_key`, then evaluate again.
```

Count is in the hash deliberately: a deduplication rebuild leaves the chunking
config identical, so nothing else would move. A refusal, not a warning — a
warning in a batch run is a line nobody reads above a number everybody quotes.

## The headline is not a recall number

It is whether the **top-1 similarity distribution for answerable questions
separates from the distribution for negative controls**. If they overlap, the
index cannot tell a question it can answer from one it cannot, and every recall
figure above it is describing noise.

Printed first, both distributions side by side with min/median/max **and every
individual score**, never collapsed into one aggregate.

`dedup_diagnostic` is excluded from both sides: q26 is a boilerplate probe and
would score high for reasons that say nothing about answerability.

## Scoring by type

| type | scored on |
|---|---|
| `single_fact` | recall@5/@20 (target chunk in top-k), top-1 correct, citation correct, rank |
| `cross_company` | the SET — **every** listed accession must appear; partial is a miss |
| `period_over_period` | **accession coverage** — both accessions must appear; target-chunk recall reported as a secondary metric |
| `negative_control` | no recall; the top-1 score IS the measurement |
| `dedup_diagnostic` | no recall; distinct accessions and distinct chunk texts in top-20 |

**Citation is `(accession, document)`, not accession alone.** q27's ground truth
lives in an exhibit whose filing's primary is a one-sentence shell; accession-only
scoring could not tell them apart, which is exactly what the exhibit expansion
was built to make measurable.

**Accession coverage is the period_over_period metric because text matching would
lie.** q19's quote is verbatim in all eight AGNC 10-Qs — a text-matching scorer
would mark a retriever correct for returning chunks from six filings nobody asked
about.

**Target-chunk recall is secondary, and declares when it does not apply.** Where
the identifying quote occurs in more than one accession corpus-wide, which
near-identical chunk ranks first turns on incidental surrounding words rather
than retrieval quality. Such entries are marked
`target_recall.meaningful: false` with `max_corpus_accessions`, rather than
omitted — an omission reads as an absent result, not an inapplicable one.

**Per-accession ranks are reported separately and never averaged.** q18's two
spans are a table row and prose; an embedder handles those very differently, so
a rank gap between its accessions is expected and is recorded as such.

## Section accuracy — reported separately, exclusions stated

Does the top hit's section match the section of the target chunk?

Scored over **BCPC and ACI only**. The output prints the exclusions and why:

| excluded | why |
|---|---|
| AGNC | chunks are unsectioned (anchor gap) |
| HON | Notes text carries `part2_item1a_risk_factors` (anchor mislabel) |
| AIG | Notes text carries `part1_item2_mdna` (anchor mislabel) |

Including them would inflate the metric: a consistently *wrong* label still
matches itself.

Measured against the **index's** label for the target chunk, not the key's
`section:` field — that field is still `TBD` for 7 of the 10 BCPC/ACI ground
truths, so scoring against it would leave the metric unscoreable exactly where
it applies.

## Procedure

1. `python -m tools.verify_key` — must report 0 flagged, and writes the binding
   record.
2. `python -m tools.evaluate`.
3. **Read the headline block first.** `distributions_overlap` is the result that
   decides whether anything below it means anything.
4. Then by-type recall, then section accuracy, then per-question detail.
5. Re-run to confirm identical scores — retrieval is deterministic for a fixed
   index, so any drift means something changed underneath.

## Edge cases

| Case | Handling |
|---|---|
| `verify_key` never run | Refused, naming the command. |
| Index rebuilt since verification | Refused, printing both fingerprints. |
| Entry has no question (`TO_WRITE`) | Logged and skipped, not scored. |
| A ground truth fails to resolve | Logged per ground truth; `unresolved_ground_truths` is recorded on the question. |
| Unknown question type | Logged, not scored, never silently treated as `single_fact`. |

## Rule notes

- **Rule 2**: wall clock, questions scored, bytes out, error per failure,
  `requests: 0`.
- **Rule 4**: a question that cannot be scored is recorded as unscored. No
  metric is imputed for it.
- **Rule 6**: the harness reports scores. It does not interpret filings, and it
  does not conclude what the numbers mean about the pipeline — that reading is
  yours.
