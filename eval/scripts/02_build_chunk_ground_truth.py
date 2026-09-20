#!/usr/bin/env python3
"""
Step 02 — Build chunk-level retrieval ground truth.

WHY THIS EXISTS
---------------
Recall is `relevant documents retrieved / relevant documents that exist`. The
denominator must be established INDEPENDENTLY of any retriever -- otherwise
every embedding model is scored against its own denominator and the numbers
are not comparable. So this script never calls an embedding model or runs a
vector search. It is pure text matching against the chunks already stored in
the production vector store.

METHOD
------
Each evaluation question carries a `reference_answer`: a verbatim passage
excerpted from its source paper (Supplementary Table S1). We locate that
passage among the chunks of that same paper.

The naive approach -- score each chunk against the answer independently --
fails badly, and the failure is instructive. Answers are longer (median ~300,
max ~1704 chars) than the 500-char chunks, so an answer routinely straddles a
chunk boundary: no single chunk then contains enough of it to look like a
match, even though the passage is plainly present in the paper. Observed on
6/50 questions, e.g. an answer beginning "Type II methanotrophs, exemplified
by Methylosinus..." whose first words land in the preceding chunk.

So instead we work the other way round:

  1. Concatenate the paper's chunks in stored order into a single text, keeping
     a character-offset map back to each chunk.
  2. Locate the answer passage in that reconstructed text (exact substring
     first, then a fuzzy alignment).
  3. Mark EVERY chunk whose character span overlaps the located passage.

Ground truth is therefore one-to-MANY by construction: a question may have
several correct chunks, which is the honest representation -- a retriever that
surfaces any chunk carrying part of the answer has done useful work.

Every question records how it was matched so match quality stays auditable:
  exact     -- answer found verbatim in the reconstructed paper text
  fuzzy     -- best alignment >= --fuzzy-threshold
  unmatched -- no alignment cleared the bar; EXCLUDED from the metric rather
               than silently counted as a miss

An unverified ground truth is the most dangerous failure mode in retrieval
evaluation: every downstream number looks rigorous while resting on a bad
denominator. Run 03_verify_ground_truth.py and inspect a sample before
trusting these numbers.

Usage:
    python3 eval/scripts/02_build_chunk_ground_truth.py
    python3 eval/scripts/02_build_chunk_ground_truth.py --fuzzy-threshold 0.75
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from difflib import SequenceMatcher

import dotenv
from pymongo import MongoClient

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv.load_dotenv(os.path.join(REPO, ".env"))

SRC_DB = os.getenv("PANKB_LLM_DATABASE", "pankb_llm")
SRC_COLLECTION = "pankb_vector_store"


def normalize(text):
    """Fold away differences that are irrelevant to whether two passages match."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def normalize_doi(d):
    d = (d or "").lower().strip()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    return d.rstrip(".").strip()


def build_paper_text(chunks):
    """
    Concatenate chunks into one text, recording each chunk's character span so a
    passage located in the concatenation can be mapped back to the chunks that
    carry it. Chunks overlap by ~100 chars, so spans overlap too -- that is
    fine and in fact desirable here.
    """
    parts, spans, cursor = [], [], 0
    for c in chunks:
        n = normalize(c["text"])
        if not n:
            continue
        parts.append(n)
        spans.append({"chunk_id": c["chunk_id"], "start": cursor, "end": cursor + len(n)})
        cursor += len(n) + 1  # +1 for the joining space
    return " ".join(parts), spans


def locate_fuzzy(answer_norm, paper_norm, threshold):
    """
    Find the window of paper text most similar to the answer. Scans at a coarse
    stride, then refines around the best hit -- exhaustive char-by-char search
    over a 100k-char paper would be far too slow.
    """
    n, m = len(paper_norm), len(answer_norm)
    if n == 0 or m == 0:
        return None
    if m >= n:
        r = SequenceMatcher(None, answer_norm, paper_norm).ratio()
        return (0, n, r) if r >= threshold else None

    stride = max(1, m // 4)
    best = (0, 0.0)
    for start in range(0, n - m + 1, stride):
        r = SequenceMatcher(None, answer_norm, paper_norm[start:start + m]).ratio()
        if r > best[1]:
            best = (start, r)

    # Refine around the coarse winner.
    lo = max(0, best[0] - stride)
    hi = min(n - m, best[0] + stride)
    fine = max(1, stride // 8)
    for start in range(lo, hi + 1, fine):
        r = SequenceMatcher(None, answer_norm, paper_norm[start:start + m]).ratio()
        if r > best[1]:
            best = (start, r)

    if best[1] < threshold:
        return None
    return (best[0], best[0] + m, best[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuzzy-threshold", type=float, default=0.70,
                    help="minimum alignment similarity to accept a fuzzy match")
    ap.add_argument("--out", default=os.path.join(REPO, "eval", "results", "ground_truth.json"))
    args = ap.parse_args()

    questions = json.load(open(os.path.join(REPO, "eval", "questions.json"), encoding="utf-8"))

    conn = os.getenv("MONGODB_CONN_STRING")
    if not conn:
        sys.exit("ERROR: MONGODB_CONN_STRING not set (expected in .env)")
    client = MongoClient(conn, serverSelectionTimeoutMS=20000)
    col = client[SRC_DB][SRC_COLLECTION]

    # Pull every chunk belonging to a source paper of the eval set. Read-only.
    eval_dois = sorted({q["source_doi"] for q in questions})
    print(f"loading chunks for {len(eval_dois)} source papers from "
          f"{SRC_DB}.{SRC_COLLECTION} (read-only) ...")
    by_doi = {}
    for doc in col.find({"source": {"$in": eval_dois}},
                        {"textContent": 1, "source": 1, "title": 1}):
        by_doi.setdefault(normalize_doi(doc["source"]), []).append({
            "chunk_id": str(doc["_id"]),
            "text": doc.get("textContent", ""),
            "title": doc.get("title", ""),
        })
    print(f"  loaded {sum(len(v) for v in by_doi.values()):,} chunks "
          f"across {len(by_doi)} papers")

    # Reconstruct each paper once; reused across its questions.
    paper_index = {}
    for doi, chunks in by_doi.items():
        text, spans = build_paper_text(chunks)
        paper_index[doi] = {"text": text, "spans": spans, "n_chunks": len(chunks)}

    out, stats = [], {"exact": 0, "fuzzy": 0, "unmatched": 0, "no_answer": 0}
    for q in questions:
        doi = normalize_doi(q["source_doi"])
        paper = paper_index.get(doi)
        answer = q.get("reference_answer", "")

        rec = {
            "id": q["id"],
            "question": q["question"],
            "source_doi": q["source_doi"],
            "n_chunks_in_paper": paper["n_chunks"] if paper else 0,
            "relevant_chunk_ids": [],
            "match_method": None,
            "match_score": None,
            "excluded": False,
            "exclusion_reason": None,
        }

        if not answer.strip():
            # Q26 carries no reference answer in Table S1 -> cannot be scored.
            rec.update(excluded=True, exclusion_reason="no_reference_answer",
                       match_method="none")
            stats["no_answer"] += 1
            out.append(rec)
            continue
        if not paper:
            rec.update(excluded=True, exclusion_reason="source_paper_not_in_store",
                       match_method="none")
            stats["unmatched"] += 1
            out.append(rec)
            continue

        a_norm = normalize(answer)
        pos = paper["text"].find(a_norm)
        if pos >= 0:
            span, method, score = (pos, pos + len(a_norm)), "exact", 1.0
            stats["exact"] += 1
        else:
            hit = locate_fuzzy(a_norm, paper["text"], args.fuzzy_threshold)
            if hit:
                span, method, score = (hit[0], hit[1]), "fuzzy", round(hit[2], 4)
                stats["fuzzy"] += 1
            else:
                rec.update(excluded=True, exclusion_reason="no_text_match",
                           match_method="none")
                stats["unmatched"] += 1
                out.append(rec)
                continue

        # Every chunk overlapping the located passage is relevant.
        relevant = [s["chunk_id"] for s in paper["spans"]
                    if s["start"] < span[1] and s["end"] > span[0]]
        rec["relevant_chunk_ids"] = relevant
        rec["match_method"] = method
        rec["match_score"] = score
        rec["answer_span"] = {"start": span[0], "end": span[1]}
        if not relevant:
            rec.update(excluded=True, exclusion_reason="no_chunk_overlap")
            stats["unmatched"] += 1
            stats[method] -= 1
        out.append(rec)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "meta": {
            "source_db": SRC_DB,
            "source_collection": SRC_COLLECTION,
            "fuzzy_threshold": args.fuzzy_threshold,
            "n_questions": len(questions),
            "stats": stats,
        },
        "questions": out,
    }
    json.dump(payload, open(args.out, "w"), indent=1)

    usable = [r for r in out if not r["excluded"]]
    n_rel = [len(r["relevant_chunk_ids"]) for r in usable]
    print("\n=== ground truth built ===")
    print(f"  exact matches : {stats['exact']}")
    print(f"  fuzzy matches : {stats['fuzzy']}")
    print(f"  unmatched     : {stats['unmatched']}  (excluded)")
    print(f"  no answer     : {stats['no_answer']}  (excluded)")
    print(f"  usable questions: {len(usable)}/{len(questions)}")
    if n_rel:
        print(f"  relevant chunks per question: min {min(n_rel)} "
              f"median {sorted(n_rel)[len(n_rel) // 2]} max {max(n_rel)}")
    print(f"\nwrote {args.out}")
    print("NEXT: run 03_verify_ground_truth.py and eyeball a sample before "
          "trusting any recall number.")


if __name__ == "__main__":
    main()
