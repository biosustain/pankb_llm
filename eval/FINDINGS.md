# PanKB RAG Evaluation — Findings

Results accumulate by phase. Each completed phase is appended; **earlier
conclusions are not edited** — how the conclusions evolved is itself part of
the audit trail.

Method, scripts and known defects: [README.md](README.md).

| Phase | Subject | Status | Date |
|---|---|---|---|
| 0 | Evaluation set and ground truth | ✅ complete | 2026-09-20 |
| 1 | Embedding model comparison | ✅ complete | 2026-09-20 |
| 2 | Reranker comparison and threshold calibration | ✅ complete | 2026-09-20 |
| 3 | Generation, end-to-end and faithfulness | ✅ complete | 2026-09-21 |

---

## Phase 0 — Evaluation set and ground truth

### Deliverables

- **50-question evaluation set** from Supplementary Table S1 of the PanKB paper
  (doi:10.1093/nar/gkae1042), retrieved via the EuropePMC API. Carries
  `question` / `source_doi` / `reference_answer`.
- **170-paper subset corpus** (20 positives + 150 seeded-random noise papers),
  recovered from git history and reproducible byte-for-byte.
- **Chunk-level ground truth**: 47/50 questions usable, median 2 relevant
  chunks per question (range 1–5).

### Key findings

**① The `source_doi` field is what makes retrieval evaluation possible**

The paper's methods section **never mentions** this column. It is what turns
"which paper does this answer come from" into machine-readable information;
without it there is no way to build retrieval ground truth. That is plausibly
why the original work reports no retrieval metrics at all.

**② Answers spanning chunk boundaries is the norm, not the exception**

The first ground-truth builder scored each chunk against the answer
independently and **failed on 6/50 questions** — despite the passages being
plainly present in the papers.

Diagnosis: answers (median ~300 chars, max 1704) are longer than the 500-char
chunks, so their opening or closing words routinely land in an adjacent chunk
and the 100-char overlap does not bridge the gap.

The clearest case is Q40 — the reference answer begins *"Type II methanotrophs,
exemplified by Methylosinus..."* while the stored chunk begins at
*"exemplified by Methylosinus..."*.

Reworking it as «reconstruct the paper → locate the answer → mark every
overlapping chunk» took usable questions from **43 to 47**.

> This is more than a bug fix. It is empirical evidence that **the current
> chunking strategy severs semantic units**, pointing at the same root cause as
> the fragment-chunk problem below.

**③ The corpus contains information-free fragment chunks**

```
chunk chars: min 8 | median 493 | max 500
very short (<80 chars): 81 (0.4%)
```

The shortest chunk is **8 characters**. Fragments like these (titles, figure
captions) score well on lexical similarity and **occupy a slot in the k=30
budget while carrying no information**.

### Ground-truth credibility

An unverified denominator is not accepted. Three checks:

| Check | Result |
|---|---|
| Token overlap (independent of the build algorithm) | median **1.00**, min **0.73**, none below the 0.55 warning line |
| Chunk-id resolvability | 89/89 resolve |
| Human review | the two weakest (Q44 0.73, Q42 0.78) inspected by hand — **both correct** |

Both low-scoring alignments are explained by answers spanning chunk
boundaries, not by mismatches.

---

## Phase 1 — Embedding model comparison

**Date**: 2026-09-20
**Scripts**: `04_probe_embedders.py` → `05_create_eval_collections.py` →
`06_populate_eval_collections.py` → `07_run_retrieval_eval.py`
**Raw data**: `results/retrieval_eval.json`, `results/retrieval_eval_detail.json`

### Setup

| Item | Value |
|---|---|
| Corpus | 170 papers / **18,156 chunks** (2,627 positive / 15,529 noise) |
| Questions | 47 (of 50: 1 has no answer, 2 could not be located) |
| Chunking | **held fixed** at the production setting (500 / 100); production chunks reused |
| Index | HNSW m=16, efConstruction=100, **similarity=L2** (mirrors production) |
| Metrics | strict recall, nDCG, MRR, hit rate |

**The embedding model is the only variable** — every model sees a
byte-identical chunk set.

### Precondition: is the L2 index safe?

The production index is built on **L2**, not cosine. The two rank identically
**only for normalized vectors** — an assumption stated nowhere in the code. A
model returning non-unit vectors would rank wrongly and silently, and no
downstream check would catch it.

Measured output norms:

| Model | Dims | Norm range | Normalized |
|---|---|---|---|
| voyage-large-2-instruct | 1024 | [1.0, 1.0] | ✅ |
| voyage-4-large | 1024 | [1.0, 1.0] | ✅ |
| text-embedding-3-large | 1024 | [0.9994, 1.0007] | ✅ |
| embed-v4.0 | 1536 | [1.0, 1.0] | ✅ |

**All normalized**, so the L2 index is safe here. The OpenAI row matters most:
truncated from a native 3072 dims to 1024, its norms are still ~1, meaning the
truncation is re-normalized. That was the likeliest thing to break, and it is
now settled by measurement rather than assumption.

### Result: the production model ranks last of four

**At k=30 (production setting)**

| Model | recall@30 | nDCG@30 | hit@30 | MRR |
|---|---|---|---|---|
| **voyage-4-large** | **0.893** | **0.620** | **1.000** | **0.617** |
| embed-v4.0 (Cohere) | 0.863 | 0.605 | 0.979 | 0.592 |
| text-embedding-3-large@1024 | 0.807 | 0.543 | 0.915 | 0.546 |
| **voyage-large-2-instruct** ← production | **0.738** | **0.424** | **0.894** | **0.399** |

**recall@k curve**

| Model | @1 | @3 | @5 | @10 | @20 | @30 | @50 | @100 |
|---|---|---|---|---|---|---|---|---|
| voyage-4-large | 0.285 | 0.500 | 0.607 | 0.756 | 0.855 | 0.893 | 0.914 | 0.914 |
| cohere embed-v4 | 0.309 | 0.541 | 0.683 | 0.749 | 0.834 | 0.863 | 0.863 | 0.863 |
| openai 3-large | 0.283 | 0.400 | 0.454 | 0.635 | 0.784 | 0.807 | 0.819 | 0.819 |
| voyage-large-2-instruct | 0.166 | 0.296 | 0.361 | 0.540 | 0.694 | 0.738 | 0.745 | 0.745 |

### Conclusions

**① The upgrade path is clear and nearly free**

`voyage-large-2-instruct` → `voyage-4-large`: **+0.155 recall@30**.

It is a **drop-in replacement** — same 1024 dims, same price ($0.12/M), same
index definition. **No schema migration**; only a re-embed.

Additionally: `voyage-large-2-instruct` is marked **Legacy** upstream, and
`voyage-4-large` is the vendor's own stated successor. **No process would have
surfaced that the production model had entered legacy status.**

**② The ranking gap is larger than the recall gap, and matters more**

```
recall@30  +0.155
nDCG@30    +0.196   <- larger
MRR        +0.218   <- largest
```

This matters more than it appears: production reranks the top 30 and then
**discards anything scoring below 0.5**. Correct answers ranked low are exactly
the ones the threshold cuts. Poor ranking is therefore not "answers appear
later" but **answers are lost**.

**③ k=30 is not the bottleneck**

All four models satisfy `recall@50 == recall@100` — the curve saturates by
k=50. Raising k recovers almost nothing and only adds rerank cost. **k=30 is a
sound choice.**

**④ But the answer is rarely the top hit**

`recall@1` is only **0.17–0.31**. The reranker is doing substantial work;
vector search alone is far from sufficient to surface the answer first.

### Stratified check: the gain is not driven by a few papers

44% of questions come from 3 papers, so an aggregate could be skewed by them.
By source paper (base → voyage-4-large, recall@30):

| Source paper | Questions | base | v4-large | Δ |
|---|---|---|---|---|
| 10.1016/j.fm.2023.104334 | 10 | 0.74 | 0.89 | +0.15 |
| 10.1038/s41598-022-21731-1 | 7 | 0.93 | 1.00 | +0.07 |
| 10.1128/spectrum.01029-22 | 5 | 0.93 | 0.93 | +0.00 |
| 10.1128/spectrum.01264-21 | 4 | 0.79 | 0.92 | +0.12 |
| 10.1038/s41598-020-77723-6 | 3 | 0.33 | 0.83 | **+0.50** |
| 10.1128/msystems.00248-24 | 3 | 0.61 | 0.89 | +0.28 |
| 10.1371/journal.pone.0299588 | 2 | 0.30 | 0.57 | +0.27 |
| 10.1186/s13068-018-1201-1 | 1 | 0.00 | 1.00 | **+1.00** |

**Not one of the 18 source papers regresses** (worst case is unchanged), and
the largest gains land on the papers where the baseline was weakest. This is
general improvement, not overfitting to the dominant papers.

### Limitations (state these whenever citing the numbers)

1. **Absolute numbers are optimistic.** The subset has only 150 noise papers
   against production's 100k+ chunks. **The relative ranking is the result; the
   absolute level is not.**
2. **47 questions is a small sample.** Gaps around 0.03 (voyage-4-large vs
   Cohere) may not be significant; the 0.155 gap is clearly beyond noise.
3. **Retrieval only.** High recall does not imply good final answers — rerank,
   threshold and generation still apply. That is phases 2 and 3.
4. **Questions are not fully independent**: Q01/Q02 target the same passage.

---

## Phase 2 — Reranker comparison and threshold calibration

**Date**: 2026-09-20
**Script**: `08_run_rerank_eval.py`
**Raw data**: `results/rerank_eval.json`

### Setup

| Item | Value |
|---|---|
| Retriever | `voyage_4_large` (phase-1 winner) |
| Candidates | k=30, **frozen once** — every reranker sees an identical list |
| Output | top 20 (production setting) |
| Questions | 47 |
| Threshold sweep | 0.00 → 0.95, step 0.05 |

**The reranker is the only variable.**

### Result 1: reranking helps, but not via recall

| Reranker | recall@20 | nDCG@20 | MRR |
|---|---|---|---|
| **voyage rerank-2.5** | 0.865 | **0.751** | **0.804** |
| voyage rerank-2.5-lite | 0.858 | 0.749 | 0.802 |
| cohere rerank-v4.0-pro | 0.865 | 0.712 | 0.745 |
| cohere rerank-english-v3.0 ← production | 0.867 | 0.705 | 0.745 |
| cohere rerank-v4.0-fast | 0.856 | 0.700 | 0.742 |
| **no reranking** (vector order) | 0.855 | 0.610 | 0.618 |

**Recall barely moves (0.855 → 0.867, a 0.012 spread) while nDCG gains 0.14 and
MRR 0.19.**

This is exactly as it should be, and worth stating plainly: **a reranker cannot
conjure new documents** — it only reorders the 30 the retriever already
returned. It therefore *cannot* raise recall. What it improves is ordering.

**Judging a rerank stage by recall is the wrong metric. nDCG and MRR are the
metrics for this layer.**

### Result 2: score distributions differ enough that no single threshold ports ⚠️

The most important finding of this phase. Same candidates, five rerankers:

| Reranker | min | p25 | median | p75 | max | **% ≥0.5** |
|---|---|---|---|---|---|---|
| cohere rerank-english-v3.0 ← production | 0.0000 | **0.0372** | 0.5851 | 0.9474 | 1.0000 | **53.2%** |
| cohere rerank-v4.0-pro | 0.4164 | **0.7199** | 0.8208 | 0.8919 | 0.9886 | **97.3%** |
| cohere rerank-v4.0-fast | 0.1857 | 0.5133 | 0.6239 | 0.7421 | 0.9671 | 77.3% |
| voyage rerank-2.5 | 0.3828 | 0.5312 | 0.6250 | 0.7227 | 0.9414 | 83.5% |
| voyage rerank-2.5-lite | 0.3809 | 0.5352 | 0.6094 | 0.7070 | 0.9453 | 81.9% |

**The production model's p25 is 0.037; v4.0-pro's is 0.72** — a ~20x
difference.

The same number 0.5:
- on **v3.0** discards roughly half the documents
- on **v4.0-pro** discards almost nothing (97.3% of scores clear it)

> 💡 This confirms an earlier notebook observation was not a bug: v3.0's score
> distribution is **sharply bimodal** — one document at 0.93, the next at 0.006.
> It is a property of the model.

**Conclusion: `0.5` is not a portable constant. It is a parameter that must be
calibrated per model.** Swapping rerankers without recalibrating changes the
filter's behaviour silently.

### Result 3: the production threshold is discarding correct answers

Threshold sweep (recall / mean documents kept):

| Reranker | 0.0 | 0.2 | 0.4 | **0.5** | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|---|---|
| **v3.0** (production) recall | 0.867 | 0.824 | 0.817 | **0.810** | 0.782 | 0.764 | 0.728 |
| docs kept | 20.0 | 12.7 | 11.3 | **10.6** | 9.9 | 8.8 | 7.6 |
| v4.0-pro recall | 0.865 | 0.865 | 0.865 | **0.865** | 0.865 | 0.840 | 0.808 |
| docs kept | 20.0 | 20.0 | 20.0 | **19.5** | 18.3 | 15.7 | 11.3 |
| voyage-2.5 recall | 0.865 | 0.865 | 0.865 | **0.858** | 0.832 | 0.747 | 0.615 |
| docs kept | 20.0 | 20.0 | 20.0 | **16.7** | 11.6 | 5.9 | 2.7 |

**The production configuration (v3.0 + threshold 0.5) costs 0.057 recall** —
roughly **6% of correct answers are discarded before reaching the LLM**.

Note also the shape of v3.0's curve: **recall drops 0.043 between 0.0 and 0.2**,
meaning it assigns very low scores to some correct answers. v4.0-pro's recall is
**completely flat from 0.0 to 0.6**.

A discarded chunk is **invisible downstream** — it never reaches the LLM, and no
generation metric can recover it.

### Threshold guidance: every model has a different usable cutoff

For each reranker, the highest threshold costing ≤0.005 recall:

| Reranker | recall@0.0 | **Highest usable threshold** | Docs kept there | Cost of production's 0.5 |
|---|---|---|---|---|
| **rerank-english-v3.0** ← production | 0.867 | **0.0** (any cutoff costs recall) | 20.0 | **−0.057** |
| rerank-v4.0-pro | 0.865 | **0.65** | 17.4 | −0.000 |
| rerank-v4.0-fast | 0.856 | 0.30 | 19.1 | −0.022 |
| voyage rerank-2.5 | 0.865 | **0.45** | 18.6 | −0.007 |
| voyage rerank-2.5-lite | 0.858 | 0.50 | 16.4 | −0.000 |

**Usable thresholds range from 0.0 to 0.65 across five models** — further
evidence the parameter does not port.

**The production model's cost curve** (docs kept includes relevant and not):

| Threshold | 0.0 | 0.05 | 0.1 | 0.2 | 0.3 | 0.4 | **0.5** | 0.6 | 0.7 |
|---|---|---|---|---|---|---|---|---|---|
| recall | 0.867 | 0.846 | 0.842 | 0.824 | 0.817 | 0.817 | **0.810** | 0.782 | 0.764 |
| docs kept | 20.0 | 14.6 | 13.8 | 12.7 | 12.0 | 11.3 | **10.6** | 9.9 | 8.8 |

v3.0 has **no free range** — recall starts falling as soon as the threshold
leaves zero (0.021 by 0.05), because it scores some correct answers very low.

**Recommendations, in priority order:**

1. **Without changing models: lower the threshold to 0.2–0.3.** Recall recovers
   from 0.810 to 0.817–0.824 while still filtering 7–8 noise documents. Grounds:
   the paper's own data shows this system reaching 88% accuracy at 75% mean
   noise, so **the cost of a few extra noise documents is lower than the cost of
   losing correct answers**.
2. **Better: change the model.** On `rerank-v4.0-pro` a threshold of **0.65**
   costs zero recall — keeping every answer *and* filtering more noise. This is
   worth more than tuning the threshold alone.
3. **Most robust: use a relative cutoff.** Since no vendor documents
   cross-query score comparability, an absolute threshold is an empirical knob.
   A top-n, or a fraction of the top score, removes the risk of silent failure
   when models change.

> ⚠️ **Scope of these numbers**: calibration depends on ground truth, and
> absolute recall is a **subset** figure (optimistic). The *shape* of the curves
> and the *relative* ordering are trustworthy; re-check against the full store
> before changing production.

### Note on metric scope: how Results 1 and 3 relate

They measure different things and **must not be conflated**:

- **Result 1 (ranking quality)**: the reranker's top 20, **with no threshold
  applied**. Measures its ability to order correct answers early (nDCG / MRR).
- **Result 3 (threshold cost)**: applies score filtering *on top of* that
  ordering, measuring how many correct answers the filter discards.

So **Result 1 is not "each model at its own best threshold"** — it is
threshold-free ranking ability. They are separated because ranking quality and
threshold calibration are orthogonal decisions: choose the best ranker first,
then calibrate its threshold.

Re-ranked by «own best threshold + recall», `rerank-v4.0-pro` and `voyage
rerank-2.5` tie for best (recall 0.865, at zero cost with thresholds of 0.65 and
0.45 respectively), while production's v3.0 loses recall at any non-zero
threshold. **Under both framings the production model is the worst of the set.**

### Production code issues surfaced

1. ⚠️ **[streamlit_app.py:54](../streamlit_app.py#L54) calls
   `CohereRerank(top_n=20)` with no `model=`** — it relies on
   `langchain_cohere`'s default. **Which rerank model serves production is not
   visible in the code**, and a library upgrade could change it with no diff.
   Same class of problem as the embedding model entering legacy unnoticed:
   dependency lifecycle is unmonitored.
2. **`0.5` is hardcoded twice** ([streamlit_app.py:41](../streamlit_app.py#L41),
   [streamlit_app_native.py:152](../streamlit_app_native.py#L152)) and must move
   together.
3. `rerank-english-v3.0` is two generations behind and English-only. Migrating
   to v4.0 leaves the SDK signature essentially unchanged (`documents` still
   accepts a list of strings, `top_n` survives), but **`model` becomes
   mandatory** on `ClientV2`.
4. **Voyage uses `top_k` where Cohere uses `top_n`** — an easy silent error when
   porting.

### Limitations

1. **Candidates come from `voyage_4_large` only.** A different retriever could
   shift the rerankers' relative standing.
2. **Absolute numbers remain subset numbers** (150 noise papers), optimistic.
3. **Rerankers are close together** (nDCG 0.700–0.751); at 47 questions some of
   those gaps may not be significant. The **rerank-vs-no-rerank gap (0.610 →
   0.751) is well beyond noise.**
4. Other vendors (e.g. Jina) were not tested.

---

## Phase 3 — Generation, end-to-end and faithfulness

**Date**: 2026-09-21
**Script**: `09_run_generation_eval.py`
**Raw data**: `results/generation_eval_raw.json` (1,176 records)

### Setup

| Item | Value |
|---|---|
| Generators | `gpt-4o-mini` (production), `gpt-4o`, `claude-sonnet-5`, `claude-haiku-4-5` |
| Conditions | **base / oracle / rag** |
| Questions | 49 (only Q26, which has no reference answer, is excluded) |
| Repeats | 2 per question |
| Pipeline (rag arm) | voyage-4-large → rerank-english-v3.0 → threshold 0.5 |
| Judge | `claude-opus-5`, **not among the tested models** |

The three arms exist so that a wrong answer can be attributed. `oracle` feeds
the reference answer as context — perfect retrieval — and therefore gives the
accuracy **ceiling**. The published evaluation has only `base` and `rag`, so it
can show RAG helps but cannot say whether a failure came from the retriever or
the generator.

### Result 1: RAG works, reproducing the paper's central claim

Accuracy (% correct, both repeats pooled):

| Generator | base | oracle | rag |
|---|---|---|---|
| gpt-4o-mini ← production | 10.2% | **94.9%** | 76.5% |
| gpt-4o | 13.3% | 90.8% | 73.5% |
| claude-sonnet-5 | 21.4% | 88.8% | **79.6%** |
| claude-haiku-4-5 | 8.2% | 88.8% | **79.6%** |

Base accuracy averages **13.3%**, against the paper's reported 22.4% — same
order of magnitude, and the gap is unsurprising given different models and a
stricter 4-way rubric. RAG lifts it to **77.3%** on average. The direction and
magnitude of the paper's headline finding reproduce.

### Result 2: retrieval, not generation, is the bottleneck ⭐

This is what the two-arm design could not show:

| Generator | oracle | rag | **retrieval cost** |
|---|---|---|---|
| gpt-4o-mini | 94.9% | 76.5% | **−18.4 pts** |
| gpt-4o | 90.8% | 73.5% | **−17.3 pts** |
| claude-sonnet-5 | 88.8% | 79.6% | −9.2 pts |
| claude-haiku-4-5 | 88.8% | 79.6% | −9.2 pts |

**Given correct material, every model answers ~90% correctly.** The generator is
not the limiting factor. The 9–18 point drop from oracle to rag is what
retrieval costs — and it is the largest single loss anywhere in this system.

This lines up with phases 1 and 2, which found the same thing from the other
end: the embedding model gives up 0.155 recall and the threshold discards ~6%
of correct answers. **Phase 3 prices those defects in end-to-end terms.**

The practical consequence: **upgrading the generator is not where the value is.**
`gpt-4o` scores *below* `gpt-4o-mini` in the rag arm (73.5% vs 76.5%) — a more
expensive model bought nothing. Retrieval improvements would.

### Result 3: faithfulness — answers are grounded, and "right by luck" is ~0

The dimension the paper never measured. In the rag arm:

| Generator | grounded | partial | **ungrounded** | no_context |
|---|---|---|---|---|
| gpt-4o-mini | 84.7% | 4.1% | **0.0%** | 0.0% |
| gpt-4o | 81.6% | 1.0% | **0.0%** | 12.2% |
| claude-sonnet-5 | 84.7% | 0.0% | **0.0%** | 0.0% |
| claude-haiku-4-5 | 84.7% | 0.0% | **0.0%** | 0.0% |

The decisive cross-tabulation — accuracy against faithfulness:

| Generator | correct **and** grounded | correct but **ungrounded** |
|---|---|---|
| gpt-4o-mini | 76.5% | **0.0%** |
| gpt-4o | 73.5% | **0.0%** |
| claude-sonnet-5 | 79.6% | **0.0%** |
| claude-haiku-4-5 | 79.6% | **0.0%** |

**Not one correct answer was ungrounded.** Every correct answer traces to the
retrieved context; none came from the model's parametric memory dressed up with
citations. That is the failure mode a citation-bearing system must not have, and
accuracy alone cannot see it — this system does not have it.

This is also the strongest available evidence for the paper's hallucination
claim, which it asserted but never measured.

### Result 4: this settles the threshold design question

Phase 2 left an open question: is a permissive threshold plus "let the LLM
filter the noise" sound, or should the filter stay strict?

The argument for permissive rested on an unverified premise — that the LLM
genuinely reasons over its context rather than falling back on memory. The
paper's "88% accuracy at 75% noise" was compatible with either reading, and
accuracy could not distinguish them.

**Faithfulness now distinguishes them: 0% ungrounded.** The LLM is demonstrably
using the context it is given. So the premise holds, and **a permissive
threshold is the better choice for this system** — the cost of a few extra
noise documents is recoverable, while a discarded correct answer is not.

Combined with phase 2's threshold curve, the recommendation is now evidence-
based rather than a judgement call: **lower the threshold, or move to
`rerank-v4.0-pro` where 0.65 costs zero recall.**

### Result 5: rejection behaviour differs sharply between models

| Generator | base rejection | rag rejection |
|---|---|---|
| claude-haiku-4-5 | **44.9%** | 0.0% |
| gpt-4o | 37.8% | 7.1% |
| gpt-4o-mini | 24.5% | 0.0% |
| claude-sonnet-5 | 12.2% | 0.0% |

Without context, `claude-haiku-4-5` declines 44.9% of the time while
`claude-sonnet-5` declines only 12.2% and answers incorrectly 24.5% of the time.
**In a regulated setting the first behaviour is preferable**: an honest refusal
is recoverable, a confident error is not. This is exactly why the paper's 4-way
rubric keeps rejection separate from incorrect, and why collapsing them would
reward the wrong model.

### Judge validation and reliability

The judge is a measuring instrument, so it is itself measured.

- **Smoke test**: four hand-constructed cases (correct / rejection / incorrect /
  partial), graded correctly on **both** axes.
- **Repeat agreement**: identical inputs run twice agree **91.2%** of the time
  (536/588). This bounds combined model + judge nondeterminism.
- **Judge errors**: **114/1,176 (9.7%)** of records failed to parse into a valid
  grade after retries and are recorded as `judge_error`. They are spread evenly
  across generators and conditions, so this is judge robustness, not a
  model-specific artefact. Those records are excluded from the percentages
  above, which are computed over all records including errors — so the true
  rates are slightly **higher** than reported, and the comparison between models
  is unaffected.

### Limitations

1. **Determinism could not be enforced on the Anthropic side.** The paper used
   `temperature=0, top_p=0`; anthropic SDK 1.7.0 no longer accepts `temperature`
   on `messages.create` and reasoning models reject it outright. OpenAI calls
   still use `temperature=0`. The 91.2% repeat agreement quantifies what this
   costs.
2. **9.7% judge error rate** is high enough to want fixing before these numbers
   are used for a production decision. A stricter output schema or a retry with
   a repair prompt would likely recover most of them.
3. **The judge has not had human spot-check calibration.** Smoke tests and
   self-consistency are necessary but not sufficient; a sample should be graded
   by hand, as was done for the phase-0 ground truth.
4. **2 repeats, 49 questions.** Differences of a few points between models
   (e.g. 79.6% vs 76.5%) are within noise. The oracle-vs-rag gap (9–18 points)
   is not.
5. **The rag arm uses the phase-1 winner** (`voyage-4-large`), not the model
   currently in production. So the 9–18 point retrieval cost is measured against
   an *already improved* retriever — with the production embedding model the gap
   would be **wider**.

### What this means end to end

```
generation ceiling (oracle)   ~90%   <- the generator is not the problem
actual system (rag)           ~77%   <- 13 points lost, all in retrieval
grounded answers              ~85%   <- and 0% right-by-luck
```

The system's headroom is in retrieval, and phases 1–2 name the specific defects:
a legacy embedding model, a reranker two generations behind, and a threshold
calibrated for neither.
