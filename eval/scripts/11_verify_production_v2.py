#!/usr/bin/env python3
"""
Step 11 — Verify the upgraded pipeline on the FULL production corpus.

Everything in phases 1-2 was measured on a 170-paper subset, which is
optimistic by construction: 18k chunks of distractors rather than 107k. This
re-runs the same comparison against the real store so the recommendation rests
on production-scale numbers, not extrapolation.

Two complete pipelines, each end to end as it would actually run:

  old   pankb_vector_store     voyage-large-2-instruct
        -> cohere rerank-english-v3.0 -> threshold 0.50
  new   pankb_vector_store_v2  voyage-4-large
        -> voyage rerank-2.5          -> threshold 0.45

Both are read-only. The comparison is deliberately of whole configurations,
not single components: the threshold is calibrated per reranker, so pairing a
new reranker with the old cutoff would measure neither configuration.

Metrics are computed after the threshold filter, which is what the LLM
actually receives -- a chunk discarded there is invisible downstream and no
generation metric can recover it.

Usage:
    python3 eval/scripts/11_verify_production_v2.py
    python3 eval/scripts/11_verify_production_v2.py --limit 10
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import dotenv
from pymongo import MongoClient

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv.load_dotenv(os.path.join(REPO, ".env"))

PROD_DB = os.getenv("PANKB_LLM_DATABASE", "pankb_llm")
RETRIEVE_K, TOP_N = 30, 20

CONFIGS = {
    "old": {
        "collection": "pankb_vector_store", "embed": "voyage-large-2-instruct",
        "embed_kw": {}, "rerank_vendor": "cohere",
        "rerank": "rerank-english-v3.0", "threshold": 0.50,
        "id_field": "_id",  # the live store predates source_id
    },
    "new": {
        "collection": "pankb_vector_store_v2", "embed": "voyage-4-large",
        "embed_kw": {"output_dimension": 1024}, "rerank_vendor": "voyage",
        "rerank": "rerank-2.5", "threshold": 0.45,
        "id_field": "source_id",  # traces back to the live store's _id
    },
}


def embed_query(cfg, text):
    import voyageai
    return voyageai.Client().embed(texts=[text], model=cfg["embed"],
                                   input_type="query",
                                   **cfg["embed_kw"]).embeddings[0]


def rerank(cfg, query, docs, attempts=4):
    for i in range(attempts):
        try:
            if cfg["rerank_vendor"] == "cohere":
                import cohere
                r = cohere.ClientV2().rerank(model=cfg["rerank"], query=query,
                                             documents=docs, top_n=TOP_N)
                return [(x.index, x.relevance_score) for x in r.results]
            import voyageai
            # Voyage uses top_k where Cohere uses top_n.
            r = voyageai.Client().rerank(query=query, documents=docs,
                                         model=cfg["rerank"], top_k=TOP_N)
            return [(x.index, x.relevance_score) for x in r.results]
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(2 ** i)


def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def score(ids, relevant):
    rel = set(relevant)
    n_rel = len(rel)
    hits = sum(1 for c in ids if c in rel)
    ideal = dcg([1.0] * min(n_rel, max(len(ids), 1)))
    first = next((i + 1 for i, c in enumerate(ids) if c in rel), None)
    return {
        "recall": hits / n_rel if n_rel else 0.0,
        "ndcg": (dcg([1.0 if c in rel else 0.0 for c in ids]) / ideal) if ideal else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        "hit": 1.0 if hits else 0.0,
        "n_kept": len(ids),
    }


def mean(v):
    return sum(v) / len(v) if v else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only N questions (smoke test)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(REPO, "eval", "results"))
    args = ap.parse_args()

    gt = json.load(open(os.path.join(REPO, "eval", "results", "ground_truth.json"),
                        encoding="utf-8"))
    questions = [q for q in gt["questions"] if not q["excluded"]]
    if args.limit:
        questions = questions[:args.limit]

    conn = os.getenv("MONGODB_CONN_STRING")
    if not conn:
        sys.exit("ERROR: MONGODB_CONN_STRING not set (expected in .env)")
    db = MongoClient(conn, serverSelectionTimeoutMS=30000)[PROD_DB]

    for name, cfg in CONFIGS.items():
        n = db[cfg["collection"]].count_documents({})
        print(f"  {name}: {cfg['collection']} ({n:,} chunks) "
              f"{cfg['embed']} -> {cfg['rerank']} >= {cfg['threshold']}")
        if n == 0:
            sys.exit(f"ERROR: {cfg['collection']} is empty")
    print(f"\nquestions: {len(questions)} (full production corpus)\n")

    results, per_q = {}, defaultdict(dict)
    for name, cfg in CONFIGS.items():
        col = db[cfg["collection"]]

        def one(q):
            vec = embed_query(cfg, q["question"])
            docs = list(col.aggregate([
                {"$search": {"cosmosSearch": {"vector": vec, "path": "vectorContent",
                                              "k": RETRIEVE_K},
                             "returnStoredSource": True}},
                {"$project": {"textContent": 1, "source_id": 1}}]))
            if not docs:
                return q["id"], score([], q["relevant_chunk_ids"]), 0
            order = rerank(cfg, q["question"], [d.get("textContent", "") for d in docs])
            # Ground truth refers to the live store's _id, so map back to it.
            kept = [str(docs[i].get(cfg["id_field"], docs[i]["_id"]))
                    for i, s in order if s >= cfg["threshold"]]
            return q["id"], score(kept, q["relevant_chunk_ids"]), len(docs)

        t0 = time.time()
        rows = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(one, q) for q in questions]):
                qid, s, n_cand = fut.result()
                rows.append(s)
                per_q[qid][name] = round(s["recall"], 4)

        agg = {
            "config": cfg, "n_questions": len(rows),
            "recall": round(mean([r["recall"] for r in rows]), 4),
            "ndcg": round(mean([r["ndcg"] for r in rows]), 4),
            "mrr": round(mean([r["mrr"] for r in rows]), 4),
            "hit": round(mean([r["hit"] for r in rows]), 4),
            "mean_docs_to_llm": round(mean([r["n_kept"] for r in rows]), 2),
            "pct_empty_context": round(100 * mean([1.0 if r["n_kept"] == 0 else 0.0
                                                   for r in rows]), 1),
            "seconds": round(time.time() - t0, 1),
        }
        results[name] = agg
        print(f"  {name:<4} recall={agg['recall']:.3f}  ndcg={agg['ndcg']:.3f}  "
              f"mrr={agg['mrr']:.3f}  docs->LLM={agg['mean_docs_to_llm']:.1f}  "
              f"({agg['seconds']:.0f}s)")

    o, n = results["old"], results["new"]
    print("\n=== full-corpus comparison (after threshold filtering) ===")
    print("  " + "metric".ljust(18) + "old".rjust(9) + "new".rjust(9) + "delta".rjust(10))
    for k, label in [("recall", "recall"), ("ndcg", "nDCG"), ("mrr", "MRR"),
                     ("hit", "hit rate"), ("mean_docs_to_llm", "docs to LLM")]:
        print("  " + label.ljust(18) + f"{o[k]:>9.3f}{n[k]:>9.3f}{n[k] - o[k]:>+10.3f}")

    improved = sum(1 for v in per_q.values() if v.get("new", 0) > v.get("old", 0))
    worse = sum(1 for v in per_q.values() if v.get("new", 0) < v.get("old", 0))
    print(f"\n  per question: {improved} improved, {worse} regressed, "
          f"{len(per_q) - improved - worse} unchanged")

    json.dump({"n_questions": len(questions), "retrieve_k": RETRIEVE_K,
               "top_n": TOP_N, "results": results, "per_question": per_q},
              open(os.path.join(args.out, "production_v2_verification.json"), "w"),
              indent=1)
    print(f"\nwrote {os.path.join(args.out, 'production_v2_verification.json')}")


if __name__ == "__main__":
    main()
