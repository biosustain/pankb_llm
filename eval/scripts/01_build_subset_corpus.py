#!/usr/bin/env python3
"""
Step 01 — Build the evaluation subset corpus.

Reconstructs the 170-paper evaluation corpus from git history:
  - 20 "positive" papers: the source papers of the 50 evaluation questions
    (Supplementary Table S1 of the PanKB paper, doi:10.1093/nar/gkae1042)
  - 150 "noise" papers: a seeded random sample of the remaining bibliome,
    acting as distractors so retrieval metrics have discriminative power.

The full 833-paper bibliome lives on `origin/develop` under Paper_all/ (it was
deleted on develop_incremental_paper_updates, so it is only reachable from that
branch). Nothing is downloaded: the corpus is recovered from the repo itself.

The noise sample is drawn with a fixed seed so that re-running this script
reproduces exactly the same corpus -- the evaluation set must be stable across
runs for results to be comparable.

Usage:
    python3 eval/scripts/01_build_subset_corpus.py [--seed 20260920] [--noise 150]
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS_BRANCH = "origin/develop"
CORPUS_PATH = "Paper_all"


def git(*args):
    r = subprocess.run(["git"] + list(args), capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def normalize_doi(d):
    """Strip scheme/host and trailing punctuation so DOIs compare reliably."""
    d = (d or "").lower().strip()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    return d.rstrip(".").strip()


def list_corpus_files():
    out = git("ls-tree", "-r", "--name-only", CORPUS_BRANCH, "--", CORPUS_PATH)
    return [p for p in out.split() if p.endswith(".txt")]


def read_paper(path):
    """Papers are stored as: line0 = DOI, line1 = title, line2+ = body."""
    txt = git("show", f"{CORPUS_BRANCH}:{path}")
    lines = txt.split("\n")
    return {
        "text": txt,
        "doi": lines[0].strip() if lines else "",
        "title": lines[1].strip() if len(lines) > 1 else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260920,
                    help="RNG seed for the noise sample (fixed for reproducibility)")
    ap.add_argument("--noise", type=int, default=150, help="number of noise papers")
    ap.add_argument("--out", default=os.path.join(REPO, "eval", "corpus"))
    args = ap.parse_args()

    questions_path = os.path.join(REPO, "eval", "questions.json")
    questions = json.load(open(questions_path, encoding="utf-8"))
    eval_dois = sorted({normalize_doi(q["source_doi"]) for q in questions})
    print(f"eval questions: {len(questions)} | unique source papers: {len(eval_dois)}")

    files = list_corpus_files()
    print(f"bibliome papers on {CORPUS_BRANCH}: {len(files)}")
    if not files:
        sys.exit(f"ERROR: no papers found at {CORPUS_BRANCH}:{CORPUS_PATH}/")

    # Index the whole bibliome by normalized DOI.
    papers, doi_to_pmid = {}, {}
    for path in files:
        pmid = os.path.basename(path)[: -len(".txt")]
        p = read_paper(path)
        papers[pmid] = p
        nd = normalize_doi(p["doi"])
        if nd:
            doi_to_pmid.setdefault(nd, pmid)

    positives, missing = [], []
    for d in eval_dois:
        (positives.append(doi_to_pmid[d]) if d in doi_to_pmid else missing.append(d))
    print(f"positives matched: {len(positives)}/{len(eval_dois)}")
    for m in missing:
        print(f"  MISSING (no full text in bibliome): {m}")

    pos_set = set(positives)
    candidates = sorted(p for p in papers if p not in pos_set)
    rng = random.Random(args.seed)
    noise = sorted(rng.sample(candidates, min(args.noise, len(candidates))))
    assert not (set(noise) & pos_set), "noise sample leaked a positive paper"

    outdir = os.path.join(args.out, "papers_subset")
    os.makedirs(outdir, exist_ok=True)
    manifest = []
    for pmid in sorted(pos_set) + noise:
        p = papers[pmid]
        with open(os.path.join(outdir, f"{pmid}.txt"), "w", encoding="utf-8") as f:
            f.write(p["text"])
        manifest.append({
            "pmid": pmid,
            "doi": p["doi"],
            "title": p["title"],
            "role": "positive" if pmid in pos_set else "noise",
            "chars": len(p["text"]),
        })

    meta = {
        "source_branch": CORPUS_BRANCH,
        "source_path": CORPUS_PATH,
        "seed": args.seed,
        "n_positive": len(pos_set),
        "n_noise": len(noise),
        "n_total": len(manifest),
        "missing_eval_dois": missing,
    }
    json.dump(manifest, open(os.path.join(args.out, "subset_manifest.json"), "w"), indent=1)
    json.dump(meta, open(os.path.join(args.out, "subset_meta.json"), "w"), indent=1)

    total_chars = sum(m["chars"] for m in manifest)
    print(f"\nwrote {len(manifest)} papers -> {outdir}")
    print(f"  positives: {len(pos_set)} | noise: {len(noise)}")
    print(f"  total chars: {total_chars:,} (~{total_chars // 500:,} chunks at 500 chars)")


if __name__ == "__main__":
    main()
