#!/usr/bin/env python3
"""
Step 07 — Run retrieval evaluation across the embedding models.

For every usable question, embeds the query with each model, runs vector search
against that model's collection, and scores the returned chunk_ids against the
step-02 ground truth. No LLM, no judge, no cost beyond query embeddings: the
scoring is arithmetic over chunk ids.

METRICS
-------
recall@k   relevant retrieved / relevant that exist. The headline number.
           Strict, not hit-rate: a question with 3 relevant chunks scores
           0.33 / 0.67 / 1.00, which is where the discriminative power lives.
nDCG@k     rewards ranking relevant chunks early. Two models with identical
           recall are not equivalent if one ranks the answer 1st and the other
           25th -- a downstream reranker or a top-n cutoff will treat them very
           differently.
MRR        reciprocal rank of the FIRST relevant chunk; "how fast is a lead
           found".
hit@k      did anything relevant come back at all. Binary and blunt, reported
           only because it answers the operational question directly.

Recall is computed at several k so the curve, not a single point, informs the
production k. If recall@30 ~ recall@100 then k=30 is already saturated and
raising it only adds rerank cost; if they diverge sharply, k=30 is discarding
answers.

Results are ALSO broken down per source paper. The question set draws 44% of
its questions from 3 papers, so an aggregate score partly measures fit to
those papers; a model that suits them looks better than it is.

Usage:
    python3 eval/scripts/07_run_retrieval_eval.py
    python3 eval/scripts/07_run_retrieval_eval.py --only voyage_4_large
    python3 eval/scripts/07_run_retrieval_eval.py --k-values 10,30,50,100
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
DEFAULT_KS = [1, 3, 5, 10, 20, 30, 50, 100]

# Must mirror step 06 exactly; a query embedded differently from the documents
# it searches is not a valid comparison.
MODEL_ARGS = {
    "voyage_large_2_instruct": {},
    "voyage_4_large": {"output_dimension": 1024},
    "openai_3_large_1024": {"dimensions": 1024},
    "cohere_embed_v4": {},
}


def embed_query_voyage(cfg, text):
    import voyageai
    kwargs = {"texts": [text], "model": cfg["model"], "input_type": "query"}
    if cfg.get("output_dimension"):
        kwargs["output_dimension"] = cfg["output_dimension"]
    return voyageai.Client().embed(**kwargs).embeddings[0]


def embed_query_openai(cfg, text):
    from openai import OpenAI
    kwargs = {"input": [text], "model": cfg["model"]}
    if cfg.get("dimensions"):
        kwargs["dimensions"] = cfg["dimensions"]
    return OpenAI().embeddings.create(**kwargs).data[0].embedding


def embed_query_cohere(cfg, text):
    import cohere
    r = cohere.ClientV2().embed(texts=[text], model=cfg["model"],
                                input_type="search_query",
                                embedding_types=["float"])
    return r.embeddings.float_[0]


QUERY_EMBEDDERS = {
    "voyage": embed_query_voyage,
    "openai": embed_query_openai,
    "cohere": embed_query_cohere,
}


def embed_with_retry(cfg, text, attempts=5):
    for i in range(attempts):
        try:
            return QUERY_EMBEDDERS[cfg["provider"]](cfg, text)
        except Exception as e:
            if i == attempts - 1:
                raise
            time.sleep(2 ** i)


def search(col, vector, k):
    """Return chunk_ids in rank order."""
    pipeline = [
        {"$search": {"cosmosSearch": {"vector": vector, "path": "vectorContent",
                                      "k": k},
                     "returnStoredSource": True}},
        {"$project": {"chunk_id": 1, "source": 1,
                      "score": {"$meta": "searchScore"}}},
    ]
    return [d.get("chunk_id") for d in col.aggregate(pipeline)]


def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def score_one(retrieved, relevant, ks):
    """All metrics for a single question."""
    rel = set(relevant)
    n_rel = len(rel)
    out = {}

    first_rank = next((i + 1 for i, c in enumerate(retrieved) if c in rel), None)
    out["mrr"] = 1.0 / first_rank if first_rank else 0.0
    out["first_relevant_rank"] = first_rank

    for k in ks:
        top = retrieved[:k]
        hits = sum(1 for c in top if c in rel)
        out[f"recall@{k}"] = hits / n_rel if n_rel else 0.0
        out[f"hit@{k}"] = 1.0 if hits else 0.0
        ideal = dcg([1.0] * min(n_rel, k))
        out[f"ndcg@{k}"] = (dcg([1.0 if c in rel else 0.0 for c in top]) / ideal
                            if ideal else 0.0)
    return out


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="evaluate a single collection")
    ap.add_argument("--k-values", default=",".join(map(str, DEFAULT_KS)))
    ap.add_argument("--out", default=os.path.join(REPO, "eval", "results"))
    args = ap.parse_args()

    ks = sorted({int(x) for x in args.k_values.split(",") if x.strip()})
    max_k = max(ks)

    gt = json.load(open(os.path.join(REPO, "eval", "results", "ground_truth.json"),
                        encoding="utf-8"))
    questions = [q for q in gt["questions"] if not q["excluded"]]
    if not questions:
        sys.exit("ERROR: ground truth has no usable questions.")

    probe = json.load(open(os.path.join(REPO, "eval", "results",
                                        "embedder_probe.json"), encoding="utf-8"))
    models = [r for r in probe["results"] if r.get("ok") and r.get("within_cosmos_limit")]
    if args.only:
        models = [m for m in models if m["collection"] == args.only]
        if not models:
            sys.exit(f"ERROR: no probed model with collection '{args.only}'")

    conn = os.getenv("MONGODB_CONN_STRING")
    if not conn:
        sys.exit("ERROR: MONGODB_CONN_STRING not set (expected in .env)")
    client = MongoClient(conn, serverSelectionTimeoutMS=30000)
    db = client[EVAL_DB]

    print(f"questions: {len(questions)} | models: {len(models)} | k up to {max_k}\n")

    all_runs, per_question = {}, defaultdict(dict)
    for m in models:
        name = m["collection"]
        col = db[name]
        n_docs = col.estimated_document_count()
        if n_docs == 0:
            print(f"  SKIP {name}: collection is empty — run step 06 first")
            continue

        cfg = {**m, **MODEL_ARGS.get(name, {})}
        print(f"  {name:<26} ({n_docs:,} docs) ", end="", flush=True)

        t0 = time.time()
        scores, by_paper = [], defaultdict(list)
        for q in questions:
            vec = embed_with_retry(cfg, q["question"])
            retrieved = search(col, vec, max_k)
            s = score_one(retrieved, q["relevant_chunk_ids"], ks)
            s["id"] = q["id"]
            s["n_relevant"] = len(q["relevant_chunk_ids"])
            s["n_retrieved"] = len(retrieved)
            scores.append(s)
            by_paper[q["source_doi"]].append(s)
            per_question[q["id"]][name] = {
                "recall@30": round(s.get("recall@30", 0.0), 4),
                "first_relevant_rank": s["first_relevant_rank"],
            }

        agg = {"collection": name, "model": m["model"], "provider": m["provider"],
               "dimensions": m["dimensions"], "n_questions": len(scores),
               "n_documents": n_docs, "seconds": round(time.time() - t0, 1),
               "mrr": round(mean([s["mrr"] for s in scores]), 4)}
        for k in ks:
            agg[f"recall@{k}"] = round(mean([s[f"recall@{k}"] for s in scores]), 4)
            agg[f"ndcg@{k}"] = round(mean([s[f"ndcg@{k}"] for s in scores]), 4)
            agg[f"hit@{k}"] = round(mean([s[f"hit@{k}"] for s in scores]), 4)

        # Per-paper, because 3 papers supply 44% of the questions.
        agg["per_paper"] = {
            doi: {"n_questions": len(rows),
                  "recall@30": round(mean([r.get("recall@30", 0.0) for r in rows]), 4)}
            for doi, rows in sorted(by_paper.items())
        }
        all_runs[name] = agg
        all_runs[name]["_per_question"] = scores
        print(f"recall@30={agg.get('recall@30'):.3f}  "
              f"ndcg@30={agg.get('ndcg@30'):.3f}  mrr={agg['mrr']:.3f}  "
              f"({agg['seconds']:.0f}s)")

    if not all_runs:
        sys.exit("\nERROR: no populated collections to evaluate.")

    # ---- report ----------------------------------------------------------
    print("\n=== recall@k ===")
    hdr = "  " + "model".ljust(26) + "".join(f"{('@' + str(k)):>9}" for k in ks)
    print(hdr)
    for name, a in all_runs.items():
        print("  " + name.ljust(26) +
              "".join(f"{a[f'recall@{k}']:>9.3f}" for k in ks))

    print("\n=== nDCG@k ===")
    print(hdr)
    for name, a in all_runs.items():
        print("  " + name.ljust(26) +
              "".join(f"{a[f'ndcg@{k}']:>9.3f}" for k in ks))

    print("\n=== summary at production k=30 ===")
    print("  " + "model".ljust(26) + "recall".rjust(8) + "ndcg".rjust(8)
          + "hit".rjust(8) + "mrr".rjust(8))
    for name, a in sorted(all_runs.items(),
                          key=lambda x: -x[1].get("recall@30", 0)):
        print("  " + name.ljust(26)
              + f"{a.get('recall@30', 0):>8.3f}{a.get('ndcg@30', 0):>8.3f}"
              + f"{a.get('hit@30', 0):>8.3f}{a['mrr']:>8.3f}")

    out = {"k_values": ks, "n_questions": len(questions),
           "ground_truth": os.path.join("eval", "results", "ground_truth.json"),
           "runs": {k: {kk: vv for kk, vv in v.items() if kk != "_per_question"}
                    for k, v in all_runs.items()}}
    json.dump(out, open(os.path.join(args.out, "retrieval_eval.json"), "w"), indent=1)
    json.dump({"per_question": per_question,
               "detail": {k: v["_per_question"] for k, v in all_runs.items()}},
              open(os.path.join(args.out, "retrieval_eval_detail.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(args.out, 'retrieval_eval.json')}")
    print(f"      {os.path.join(args.out, 'retrieval_eval_detail.json')}")


if __name__ == "__main__":
    main()
