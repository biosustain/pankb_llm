# PanKB LLM — Retrieval & Generation Evaluation

Evaluation harness for the PanKB RAG system. Built to answer questions the
published evaluation does not: the paper reports one **end-to-end** accuracy
number (88%, 92% for the best system) and no retrieval metrics at all, so a
wrong answer cannot be attributed to the retriever or the generator.

Three layers, evaluated separately so failures can be attributed:

| Layer | Question | Metrics | Status |
|---|---|---|---|
| Retrieval | Did we find the passages that contain the answer? | recall@k, nDCG@k, MRR | ✅ complete |
| Generation | Given correct context, does the model use it? | accuracy, faithfulness | ✅ complete |
| End-to-end | How does the whole chain perform? | accuracy (4-way), faithfulness | ✅ complete |

Results for all four phases are in [FINDINGS.md](FINDINGS.md).

## Evaluation set

`questions.json` / `questions.csv` — the 50-question set from Supplementary
Table S1 of the PanKB paper (doi:10.1093/nar/gkae1042), recovered from the
EuropePMC supplementary-files API. `Table_S1_original.xlsx` is the unmodified
source file, kept for provenance.

Each question carries:

- `question` — the query
- `source_doi` — the paper the answer was taken from (**not** mentioned in the
  paper's methods, but present in the data; it is what makes retrieval
  evaluation possible at all)
- `reference_answer` — the verbatim passage from that paper

Known defects, handled explicitly rather than silently:

- **Q26 has no reference answer** in Table S1. Excluded from scoring.
- **Coverage is uneven.** The 50 questions come from only 20 papers, and the
  top 3 papers supply 22 of them (44%). Report results per source paper as
  well as in aggregate; a model that happens to suit the dominant papers will
  otherwise look better than it is.
- **Q01/Q02 target the same passage**, so questions are not fully independent.
- Answers are verbatim excerpts, up to 1704 chars, so scoring answers requires
  a judge (string equality is useless here).

Licensing: the question set is redistributable (the PanKB paper is CC-BY), but
`reference_answer` quotes 20 third-party papers that are mostly not CC-BY.
Fine as a private eval set; publishing it needs a second look.

## Corpus

`corpus/papers_subset/` — 170 papers used to build the experimental vector
stores:

- **20 positives** — the source papers of the 50 questions
- **150 noise** — seeded random sample of the rest of the bibliome, as
  distractors so retrieval metrics have discriminative power

Recovered from `origin/develop:Paper_all/` (833 papers); nothing is downloaded.
The noise sample uses a fixed seed, so the corpus is byte-identical on re-run.

A subset rather than the full 106k-chunk production store, because comparing
embedding models means re-embedding everything once per model. **This inflates
absolute recall** (fewer distractors than production) but does not bias the
*relative* ranking of models, since all models face the same corpus. Cite
absolute numbers as subset numbers.

## Scripts

Run in order from the repo root.

```bash
# Phase 0 — evaluation set and ground truth
python3 eval/scripts/01_build_subset_corpus.py        # rebuild corpus from git
python3 eval/scripts/02_build_chunk_ground_truth.py   # locate answers in chunks
python3 eval/scripts/03_verify_ground_truth.py        # check the ground truth

# Phase 1 — embedding model comparison
python3 eval/scripts/04_probe_embedders.py            # dims + norms, no writes
python3 eval/scripts/05_create_eval_collections.py    # --apply to create
python3 eval/scripts/06_populate_eval_collections.py  # --apply to embed
python3 eval/scripts/07_run_retrieval_eval.py         # recall/nDCG/MRR

# Phase 2 — reranker comparison and threshold calibration
python3 eval/scripts/08_run_rerank_eval.py

# Phase 3 — generation, end-to-end, faithfulness
python3 eval/scripts/09_run_generation_eval.py        # --apply to run

# Phase 4 — full-corpus production upgrade
python3 eval/scripts/10_build_production_v2_store.py  # --apply to build
python3 eval/scripts/11_verify_production_v2.py       # old vs new, full corpus
```

**What each script touches.** `01` only reads git. `02`, `03`, `07`, `08` and
`11` read vector stores but never write to them. `05`, `06` write only to the
`eval` database. `10` is the sole script that writes to the production
database, and only ever to a **new** collection — the live
`pankb_vector_store` is read-only throughout.

Scripts that write default to a dry run and require `--apply`. `06`, `09` and
`10` resume from where they stopped, so an interrupted run neither duplicates
documents nor pays twice.

Phases 0–3 use the 170-paper subset; phase 4 uses the full production corpus.

### Ground truth (`02`)

Recall is `relevant retrieved / relevant that exist`. The denominator must be
fixed independently of any retriever, or each model is scored against its own
denominator and the numbers cannot be compared. So `02` uses **pure text
matching** — no embedding model, no vector search.

Answers (median ~300 chars, max 1704) are longer than the 500-char chunks, so
an answer routinely straddles a chunk boundary. Scoring chunks individually
fails on exactly those cases: 6/50 questions were unmatchable that way, despite
the passage being plainly present. `02` instead reconstructs each paper from
its chunks, locates the answer in that text, and marks every chunk overlapping
the located span. Ground truth is one-to-many by construction.

Current result: **47/50 usable** — 19 exact, 28 fuzzy, 2 unmatched, 1 without a
reference answer. Median 2 relevant chunks per question (range 1–5).

#### Why one-to-many matters for the metric

The multi-chunk structure is not an inconvenience to be engineered away — it
is what makes recall a useful measurement here. With exactly one relevant
chunk per question, recall can only be 0 or 1: it collapses into accuracy, and
two models that both "found 30 questions" are indistinguishable even if one
surfaced the central passage and the other barely clipped its edge. With
several relevant chunks, recall becomes continuous (a question with 3 relevant
chunks scores 0.33 / 0.67 / 1.00), which is where the discriminative power
comes from. nDCG needs this too: with a single relevant document it degenerates
towards MRR.

We report **strict recall** — `relevant retrieved / relevant that exist` — not
a hit-rate that scores 1 whenever any relevant chunk is found. Hit-rate answers
a different, operational question ("could this question be answered at all?")
and is worth revisiting later, but it is binary and therefore blunt for
comparing embedding models.

#### On chunk size and overlap

No chunk size makes answers stop spanning boundaries. The longest reference
answer is 1704 chars; a chunk large enough to contain it whole would dilute the
embedding of every short passage, mixing several topics into one vector and
degrading retrieval precision. Small chunks localise well but sever semantics;
large chunks preserve semantics but retrieve poorly and waste context. Spanning
is the normal case, and one-to-many ground truth models it honestly rather than
trying to eliminate it.

Raising `chunk_overlap` reduces severance but is not free either: it inflates
the store and manufactures near-duplicate chunks, so a top-k can fill up with
several slicings of the same passage — recall looks higher while the context
actually carries less independent information.

These are tunable parameters and this harness could measure them, but every
change requires re-embedding the entire corpus. Deferred: the current
experiment holds chunking fixed at the production setting (500 / 100) so that
the embedding model is the only variable.

### Verification (`03`)

Unverified ground truth is the most dangerous failure mode here: every
downstream number looks rigorous while resting on a bad denominator. `03`
checks it three ways — automated sanity checks, a token-overlap confirmation
independent of the alignment used to build the truth, and a human review file
(`results/ground_truth_review.txt`) pairing each answer with its chunks.

Current result: token overlap median **1.00**, minimum **0.73**, nothing below
the 0.55 warning line, all chunk ids resolve. The two weakest alignments
(Q44 0.73, Q42 0.78) were inspected by hand and are correct — both score below
1.0 only because the answer spans a chunk boundary.

## Full-corpus verification (`10`, `11`)

Phases 0–3 run on the 170-paper subset, which is optimistic by construction.
`10` rebuilds the **entire** production corpus (106,717 chunks) with the
winning embedding model into a new collection, and `11` re-measures the old
and new pipelines end to end against it — so the recommendation rests on
production-scale numbers rather than extrapolation.

`10` re-embeds the stored `textContent` rather than re-splitting the papers.
That keeps chunk boundaries byte-identical to the live store, so the embedding
model is the only thing that changes and the subset comparison transfers. It
writes only to the new collection; the live one is untouched and remains a
zero-cost rollback.

`11` compares whole configurations rather than single components, because the
relevance threshold is calibrated per reranker — pairing a new reranker with
the old cutoff would measure neither. Metrics are computed **after** threshold
filtering, i.e. on what the LLM actually receives: a chunk discarded there is
invisible downstream and no generation metric can recover it.

## Results

`results/` holds generated artifacts (ground truth, verification reports,
retrieval runs). They are committed deliberately: a metric is only meaningful
alongside the exact question set, corpus and ground truth it was computed
against.

## Notes on the vector store

- **Azure Cosmos DB for MongoDB vCore caps vectors at 2000 dimensions.** This
  constrains model choice more than quality does.
- The production HNSW index is built with `similarity: "L2"`, not cosine. These
  rank identically **only for normalized vectors** — an assumption that holds
  for the current model but is written down nowhere in the code. Any candidate
  embedding model must have its output norms checked before its numbers are
  trusted, and dimension-truncated vectors especially so.
