#!/usr/bin/env python3
"""
Step 03 — Verify the chunk-level ground truth.

Ground truth built by automated matching can be wrong, and wrong ground truth
is the worst failure mode in retrieval evaluation: every recall number
downstream looks rigorous while resting on a bad denominator. Nothing here is
trusted until it has been checked.

Three checks:

  1. AUTOMATED SANITY
     - fuzzy matches near the acceptance threshold (most likely to be wrong)
     - questions whose relevant-chunk count looks implausible
     - relevant chunks that are suspiciously short (title fragments etc.)
     - chunk ids that don't resolve in the store

  2. TOKEN-OVERLAP CONFIRMATION
     Independent of the SequenceMatcher alignment used to build the truth: what
     fraction of the answer's distinctive tokens actually appear in the chunks
     marked relevant? A genuine match scores high. This catches alignments that
     scored well positionally but landed on the wrong passage.

  3. HUMAN SAMPLE
     Writes a review file pairing each sampled question's answer with the text
     of its chunks, for eyeballing. Automated checks cannot certify a
     denominator; a person still has to look.

Usage:
    python3 eval/scripts/03_verify_ground_truth.py
    python3 eval/scripts/03_verify_ground_truth.py --sample 15 --seed 7
"""

import argparse
import json
import os
import random
import re
import sys
import unicodedata

import dotenv
from pymongo import MongoClient

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv.load_dotenv(os.path.join(REPO, ".env"))

SRC_DB = os.getenv("PANKB_LLM_DATABASE", "pankb_llm")
SRC_COLLECTION = "pankb_vector_store"

# Ignored when measuring "distinctive" token overlap.
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are", "was",
    "were", "be", "been", "as", "by", "with", "that", "this", "these", "those",
    "it", "its", "on", "at", "from", "which", "can", "has", "have", "had",
    "also", "such", "than", "then", "but", "not", "we", "our", "their", "they",
}


def normalize(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).lower().strip()


def content_tokens(text):
    toks = re.findall(r"[a-z0-9][a-z0-9\-\.]{2,}", normalize(text))
    return {t for t in toks if t not in STOPWORDS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=12, help="questions to write for human review")
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--overlap-warn", type=float, default=0.55,
                    help="flag questions whose token overlap falls below this")
    ap.add_argument("--gt", default=os.path.join(REPO, "eval", "results", "ground_truth.json"))
    ap.add_argument("--out", default=os.path.join(REPO, "eval", "results"))
    args = ap.parse_args()

    gt = json.load(open(args.gt, encoding="utf-8"))
    questions = {q["id"]: q for q in json.load(
        open(os.path.join(REPO, "eval", "questions.json"), encoding="utf-8"))}
    records = gt["questions"]
    usable = [r for r in records if not r["excluded"]]

    conn = os.getenv("MONGODB_CONN_STRING")
    if not conn:
        sys.exit("ERROR: MONGODB_CONN_STRING not set (expected in .env)")
    col = MongoClient(conn, serverSelectionTimeoutMS=20000)[SRC_DB][SRC_COLLECTION]

    # Resolve every referenced chunk id once.
    from bson import ObjectId
    wanted = {cid for r in usable for cid in r["relevant_chunk_ids"]}
    print(f"resolving {len(wanted):,} referenced chunks (read-only) ...")
    chunk_text = {}
    for doc in col.find({"_id": {"$in": [ObjectId(c) for c in wanted]}},
                        {"textContent": 1}):
        chunk_text[str(doc["_id"])] = doc.get("textContent", "")
    missing_ids = wanted - set(chunk_text)
    print(f"  resolved {len(chunk_text):,} | unresolvable: {len(missing_ids)}")

    # ---- Check 2: token-overlap confirmation -------------------------------
    flagged, overlaps = [], []
    for r in usable:
        answer = questions[r["id"]]["reference_answer"]
        a_tok = content_tokens(answer)
        joined = " ".join(chunk_text.get(c, "") for c in r["relevant_chunk_ids"])
        c_tok = content_tokens(joined)
        ov = (len(a_tok & c_tok) / len(a_tok)) if a_tok else 0.0
        r["_token_overlap"] = round(ov, 3)
        overlaps.append(ov)
        if ov < args.overlap_warn:
            flagged.append((r["id"], round(ov, 3), r["match_method"], r.get("match_score")))

    # ---- Check 1: automated sanity ----------------------------------------
    short_chunks = [(r["id"], c, len(chunk_text.get(c, "")))
                    for r in usable for c in r["relevant_chunk_ids"]
                    if len(chunk_text.get(c, "")) < 80]
    near_thresh = [(r["id"], r["match_score"]) for r in usable
                   if r["match_method"] == "fuzzy" and r["match_score"] is not None
                   and r["match_score"] < gt["meta"]["fuzzy_threshold"] + 0.06]

    ov_sorted = sorted(overlaps)
    print("\n=== token-overlap confirmation ===")
    print(f"  questions checked: {len(usable)}")
    if ov_sorted:
        print(f"  overlap  min {ov_sorted[0]:.2f} | median "
              f"{ov_sorted[len(ov_sorted) // 2]:.2f} | max {ov_sorted[-1]:.2f}")
        print(f"  >=0.80: {sum(1 for o in overlaps if o >= 0.8)}  "
              f">=0.55: {sum(1 for o in overlaps if o >= 0.55)}  "
              f"<0.55: {sum(1 for o in overlaps if o < 0.55)}")

    print("\n=== automated sanity ===")
    print(f"  unresolvable chunk ids : {len(missing_ids)}")
    print(f"  very short chunks (<80c): {len(short_chunks)}")
    print(f"  fuzzy matches near threshold: {len(near_thresh)}")
    if flagged:
        print(f"\n  LOW OVERLAP (inspect these first): {len(flagged)}")
        for qid, ov, method, score in sorted(flagged, key=lambda x: x[1])[:10]:
            print(f"    {qid}  overlap={ov}  method={method}  align={score}")

    # ---- Check 3: human review file ---------------------------------------
    rng = random.Random(args.seed)
    priority = [r for r in usable if r["_token_overlap"] < args.overlap_warn]
    rest = [r for r in usable if r["_token_overlap"] >= args.overlap_warn]
    sample = priority + rng.sample(rest, min(max(0, args.sample - len(priority)), len(rest)))

    review_path = os.path.join(args.out, "ground_truth_review.txt")
    with open(review_path, "w", encoding="utf-8") as f:
        f.write("GROUND TRUTH — HUMAN REVIEW\n")
        f.write("For each question: does at least one chunk below actually contain\n")
        f.write("the reference answer? Mark VERDICT ok / wrong / partial.\n")
        f.write("Low-overlap questions are listed first (most likely to be wrong).\n")
        f.write("=" * 78 + "\n\n")
        for r in sample:
            q = questions[r["id"]]
            f.write(f"[{r['id']}]  match={r['match_method']} align={r.get('match_score')} "
                    f"token_overlap={r['_token_overlap']}\n")
            f.write(f"Q: {q['question']}\n\n")
            f.write(f"REFERENCE ANSWER:\n{q['reference_answer']}\n\n")
            f.write(f"CHUNKS MARKED RELEVANT ({len(r['relevant_chunk_ids'])}):\n")
            for i, cid in enumerate(r["relevant_chunk_ids"], 1):
                f.write(f"  --- chunk {i} ({cid}) ---\n  "
                        f"{chunk_text.get(cid, '<UNRESOLVED>')}\n\n")
            f.write("VERDICT: ____\n")
            f.write("-" * 78 + "\n\n")

    report = {
        "n_usable": len(usable),
        "n_excluded": len(records) - len(usable),
        "token_overlap": {
            "min": round(ov_sorted[0], 3) if ov_sorted else None,
            "median": round(ov_sorted[len(ov_sorted) // 2], 3) if ov_sorted else None,
            "max": round(ov_sorted[-1], 3) if ov_sorted else None,
            "n_below_warn": len(flagged),
        },
        "unresolvable_chunk_ids": len(missing_ids),
        "short_relevant_chunks": len(short_chunks),
        "fuzzy_near_threshold": len(near_thresh),
        "flagged_questions": [f[0] for f in flagged],
        "review_sample": [r["id"] for r in sample],
    }
    json.dump(report, open(os.path.join(args.out, "ground_truth_verification.json"), "w"), indent=1)

    print(f"\nwrote {review_path}")
    print(f"      {os.path.join(args.out, 'ground_truth_verification.json')}")
    print("\nREAD the review file before trusting any recall number.")


if __name__ == "__main__":
    main()
