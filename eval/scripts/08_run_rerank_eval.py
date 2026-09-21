#!/usr/bin/env python3
"""
Step 08 — Evaluate the rerank stage and calibrate the relevance threshold.

Three questions the published work never asked:

  1. Does reranking actually help? The live pipeline retrieves k=30 and reranks
     to top 20, but the gain was never measured. Here the no-rerank baseline is
     just the vector order, scored with the same metrics.

  2. Which reranker? Production runs `rerank-english-v3.0`, two generations
     behind `rerank-v4.0-pro` (4k vs 32k context). Voyage's rerank-2.5 family
     is included as a cross-vendor comparison.

  3. What should the threshold be? This is the important one. Production hard-
     filters reranked documents at `relevance_score >= 0.5` -- a number carried
     over from the paper, where it applied to embedding SIMILARITY, not to a
     reranker's relevance score. Those are different scales.

     Cohere documents its scores as normalized to [0,1] but publishes no
     guidance on picking a cutoff and never claims scores are comparable across
     queries; normalized range is not calibration. Voyage documents no
     normalization at all. So the threshold is an empirical, per-model knob,
     and this script measures it rather than assuming it: for each reranker it
     sweeps cutoffs and reports how many relevant chunks each one DISCARDS.

     A discarded chunk is invisible downstream -- it never reaches the LLM and
     no generation metric can recover it.

Retrieval is read from the frozen candidates produced by step 07's search, so
every reranker sees an identical candidate list and the reranker is the only
variable.

Usage:
    python3 eval/scripts/08_run_rerank_eval.py
    python3 eval/scripts/08_run_rerank_eval.py --retriever voyage_4_large
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import dotenv
from pymongo import MongoClient

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv.load_dotenv(os.path.join(REPO, ".env"))

EVAL_DB = "eval"
RETRIEVE_K = 30   # production: vector search depth
TOP_N = 20        # production: how many survive reranking
THRESHOLDS = [round(x / 20, 2) for x in range(0, 20)]  # 0.00 .. 0.95

# Query embedders, mirroring step 06/07 so documents and queries stay in the
# same space.
MODEL_ARGS = {
    "voyage_large_2_instruct": {},
    "voyage_4_large": {"output_dimension": 1024},
    "openai_3_large_1024": {"dimensions": 1024},
    "cohere_embed_v4": {},
}

RERANKERS = [
    {"key": "cohere_rerank_english_v3", "provider": "cohere",
     "model": "rerank-english-v3.0", "note": "production baseline (legacy, 4k ctx)"},
    {"key": "cohere_rerank_v4_pro", "provider": "cohere",
     "model": "rerank-v4.0-pro", "note": "Cohere flagship (32k ctx)"},
    {"key": "cohere_rerank_v4_fast", "provider": "cohere",
     "model": "rerank-v4.0-fast", "note": "Cohere low-latency (32k ctx)"},
    {"key": "voyage_rerank_2_5", "provider": "voyage",
     "model": "rerank-2.5", "note": "Voyage stable (32k ctx)"},
    {"key": "voyage_rerank_2_5_lite", "provider": "voyage",
     "model": "rerank-2.5-lite", "note": "Voyage lite (32k ctx)"},
]


def embed_query(cfg, text):
    if cfg["provider"] == "voyage":
        import voyageai
        kw = {"texts": [text], "model": cfg["model"], "input_type": "query"}
        if cfg.get("output_dimension"):
            kw["output_dimension"] = cfg["output_dimension"]
        return voyageai.Client().embed(**kw).embeddings[0]
    if cfg["provider"] == "openai":
        from openai import OpenAI
        kw = {"input": [text], "model": cfg["model"]}
        if cfg.get("dimensions"):
            kw["dimensions"] = cfg["dimensions"]
        return OpenAI().embeddings.create(**kw).data[0].embedding
    import cohere
    return cohere.ClientV2().embed(texts=[text], model=cfg["model"],
                                   input_type="search_query",
                                   embedding_types=["float"]).embeddings.float_[0]


def rerank(cfg, query, docs, top_n, attempts=5):
    """Returns [(original_index, relevance_score)] in reranked order."""
    for i in range(attempts):
        try:
            if cfg["provider"] == "cohere":
                import cohere
                r = cohere.ClientV2().rerank(model=cfg["model"], query=query,
                                             documents=docs, top_n=top_n)
                return [(x.index, x.relevance_score) for x in r.results]
            import voyageai
            # NOTE: Voyage uses top_k where Cohere uses top_n.
            r = voyageai.Client().rerank(query=query, documents=docs,
                                         model=cfg["model"], top_k=top_n)
            return [(x.index, x.relevance_score) for x in r.results]
        except Exception as e:
            if i == attempts - 1:
                raise
            time.sleep(2 ** i)


def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def score(ordered_ids, relevant, k):
    """recall/ndcg/mrr over an ordered chunk_id list."""
    rel = set(relevant)
    n_rel = len(rel)
    top = ordered_ids[:k]
    hits = sum(1 for c in top if c in rel)
    ideal = dcg([1.0] * min(n_rel, k))
    first = next((i + 1 for i, c in enumerate(top) if c in rel), None)
    return {
        "recall": hits / n_rel if n_rel else 0.0,
        "ndcg": (dcg([1.0 if c in rel else 0.0 for c in top]) / ideal) if ideal else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        "hit": 1.0 if hits else 0.0,
    }


def mean(v):
    return sum(v) / len(v) if v else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="voyage_4_large",
                    help="collection whose vector search supplies candidates")
    ap.add_argument("--out", default=os.path.join(REPO, "eval", "results"))
    args = ap.parse_args()

    gt = json.load(open(os.path.join(REPO, "eval", "results", "ground_truth.json"),
                        encoding="utf-8"))
    questions = [q for q in gt["questions"] if not q["excluded"]]

    probe = json.load(open(os.path.join(REPO, "eval", "results",
                                        "embedder_probe.json"), encoding="utf-8"))
    retr = next((r for r in probe["results"] if r["collection"] == args.retriever), None)
    if not retr:
        sys.exit(f"ERROR: unknown retriever collection '{args.retriever}'")
    retr_cfg = {**retr, **MODEL_ARGS.get(args.retriever, {})}

    conn = os.getenv("MONGODB_CONN_STRING")
    if not conn:
        sys.exit("ERROR: MONGODB_CONN_STRING not set (expected in .env)")
    col = MongoClient(conn, serverSelectionTimeoutMS=30000)[EVAL_DB][args.retriever]
    if col.estimated_document_count() == 0:
        sys.exit(f"ERROR: collection '{args.retriever}' is empty — run step 06")

    # ---- freeze the candidate set once -----------------------------------
    print(f"retriever: {args.retriever} ({retr['model']})")
    print(f"fetching k={RETRIEVE_K} candidates for {len(questions)} questions ...")
    candidates = {}
    for q in questions:
        vec = embed_query(retr_cfg, q["question"])
        docs = list(col.aggregate([
            {"$search": {"cosmosSearch": {"vector": vec, "path": "vectorContent",
                                          "k": RETRIEVE_K},
                         "returnStoredSource": True}},
            {"$project": {"chunk_id": 1, "textContent": 1,
                          "score": {"$meta": "searchScore"}}},
        ]))
        candidates[q["id"]] = [{"chunk_id": d["chunk_id"],
                                "text": d.get("textContent", "")} for d in docs]
    n_cand = mean([len(v) for v in candidates.values()])
    print(f"  {n_cand:.1f} candidates per question (frozen; every reranker "
          f"sees the same list)\n")

    runs = {}

    # ---- baseline: no reranking ------------------------------------------
    base = [score([c["chunk_id"] for c in candidates[q["id"]]],
                  q["relevant_chunk_ids"], TOP_N) for q in questions]
    runs["no_rerank"] = {
        "key": "no_rerank", "model": "(vector order, no reranking)",
        "provider": "-", "note": "baseline: first 20 of the vector results",
        f"recall@{TOP_N}": round(mean([s["recall"] for s in base]), 4),
        f"ndcg@{TOP_N}": round(mean([s["ndcg"] for s in base]), 4),
        "mrr": round(mean([s["mrr"] for s in base]), 4),
        f"hit@{TOP_N}": round(mean([s["hit"] for s in base]), 4),
        "threshold_sweep": None,  # no comparable score to threshold on
    }
    print(f"  {'no_rerank':<26} recall@{TOP_N}="
          f"{runs['no_rerank'][f'recall@{TOP_N}']:.3f}  "
          f"ndcg={runs['no_rerank'][f'ndcg@{TOP_N}']:.3f}  "
          f"mrr={runs['no_rerank']['mrr']:.3f}")

    # ---- each reranker ----------------------------------------------------
    for rr in RERANKERS:
        t0 = time.time()
        scores, sweep_rows, all_scores = [], [], []
        try:
            for q in questions:
                cand = candidates[q["id"]]
                order = rerank(rr, q["question"], [c["text"] for c in cand], TOP_N)
                ids = [cand[i]["chunk_id"] for i, _ in order]
                s = score(ids, q["relevant_chunk_ids"], TOP_N)
                scores.append(s)

                rel = set(q["relevant_chunk_ids"])
                all_scores.extend(sc for _, sc in order)
                # What each cutoff would keep / discard for THIS question.
                row = {}
                for t in THRESHOLDS:
                    kept = [(cand[i]["chunk_id"], sc) for i, sc in order if sc >= t]
                    kept_rel = sum(1 for cid, _ in kept if cid in rel)
                    row[t] = {
                        "kept": len(kept),
                        "kept_relevant": kept_rel,
                        "recall": kept_rel / len(rel) if rel else 0.0,
                        "empty": 1 if not kept else 0,
                    }
                sweep_rows.append(row)
        except Exception as e:
            print(f"  {rr['key']:<26} FAILED — {type(e).__name__}: {str(e)[:70]}")
            continue

        sweep = {}
        for t in THRESHOLDS:
            rows = [r[t] for r in sweep_rows]
            sweep[str(t)] = {
                "mean_kept": round(mean([r["kept"] for r in rows]), 2),
                "recall": round(mean([r["recall"] for r in rows]), 4),
                "pct_questions_empty": round(100 * mean([r["empty"] for r in rows]), 1),
            }

        srt = sorted(all_scores)
        runs[rr["key"]] = {
            "key": rr["key"], "model": rr["model"], "provider": rr["provider"],
            "note": rr["note"], "seconds": round(time.time() - t0, 1),
            f"recall@{TOP_N}": round(mean([s["recall"] for s in scores]), 4),
            f"ndcg@{TOP_N}": round(mean([s["ndcg"] for s in scores]), 4),
            "mrr": round(mean([s["mrr"] for s in scores]), 4),
            f"hit@{TOP_N}": round(mean([s["hit"] for s in scores]), 4),
            "score_distribution": {
                "min": round(srt[0], 6), "p25": round(srt[len(srt) // 4], 6),
                "median": round(srt[len(srt) // 2], 6),
                "p75": round(srt[3 * len(srt) // 4], 6),
                "max": round(srt[-1], 6),
                "pct_above_0.5": round(100 * mean([1.0 if s >= 0.5 else 0.0
                                                   for s in all_scores]), 1),
            },
            "threshold_sweep": sweep,
        }
        a = runs[rr["key"]]
        print(f"  {rr['key']:<26} recall@{TOP_N}={a[f'recall@{TOP_N}']:.3f}  "
              f"ndcg={a[f'ndcg@{TOP_N}']:.3f}  mrr={a['mrr']:.3f}  "
              f"({a['seconds']:.0f}s)")

    # ---- report -----------------------------------------------------------
    print(f"\n=== rerank quality (top {TOP_N} of {RETRIEVE_K} candidates) ===")
    print("  " + "reranker".ljust(26) + "recall".rjust(8) + "ndcg".rjust(8)
          + "mrr".rjust(8) + "hit".rjust(8))
    for k, a in sorted(runs.items(), key=lambda x: -x[1].get(f"ndcg@{TOP_N}", 0)):
        print("  " + k.ljust(26) + f"{a.get(f'recall@{TOP_N}', 0):>8.3f}"
              + f"{a.get(f'ndcg@{TOP_N}', 0):>8.3f}{a.get('mrr', 0):>8.3f}"
              + f"{a.get(f'hit@{TOP_N}', 0):>8.3f}")

    print("\n=== relevance score distributions (why one threshold cannot "
          "serve every model) ===")
    print("  " + "reranker".ljust(26) + "min".rjust(9) + "p25".rjust(9)
          + "median".rjust(9) + "p75".rjust(9) + "max".rjust(9) + ">=0.5".rjust(9))
    for k, a in runs.items():
        d = a.get("score_distribution")
        if not d:
            continue
        print("  " + k.ljust(26) + f"{d['min']:>9.4f}{d['p25']:>9.4f}"
              + f"{d['median']:>9.4f}{d['p75']:>9.4f}{d['max']:>9.4f}"
              + f"{d['pct_above_0.5']:>8.1f}%")

    print("\n=== what the production threshold 0.5 would do to each model ===")
    print("  " + "reranker".ljust(26) + "kept".rjust(8) + "recall".rjust(9)
          + "lost".rjust(9) + "empty".rjust(9))
    for k, a in runs.items():
        sw = a.get("threshold_sweep")
        if not sw:
            continue
        at0 = sw["0.0"]
        at5 = sw["0.5"]
        lost = at0["recall"] - at5["recall"]
        print("  " + k.ljust(26) + f"{at5['mean_kept']:>8.1f}"
              + f"{at5['recall']:>9.3f}" + f"{lost:>+9.3f}"
              + f"{at5['pct_questions_empty']:>8.1f}%")

    out = {"retriever": args.retriever, "retrieve_k": RETRIEVE_K, "top_n": TOP_N,
           "n_questions": len(questions), "thresholds": THRESHOLDS, "runs": runs}
    json.dump(out, open(os.path.join(args.out, "rerank_eval.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(args.out, 'rerank_eval.json')}")


if __name__ == "__main__":
    main()
