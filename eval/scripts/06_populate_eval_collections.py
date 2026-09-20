#!/usr/bin/env python3
"""
Step 06 — Populate each eval collection by re-embedding the same chunks.

Reads the chunk TEXT of the 170 subset papers from the production store
(read-only) and re-embeds it with each candidate model, writing the vectors
into that model's collection in the `eval` database.

Reusing production chunks rather than re-splitting the papers is the point:
chunk boundaries then match the live system exactly, and — more importantly —
every model sees a byte-identical chunk set, so the embedding model is the
only variable. Re-splitting locally would silently introduce a second one,
since the production store was built with LangChain's
RecursiveCharacterTextSplitter while make_vectordb_native.py uses a different
splitter that does not reproduce the same boundaries.

`chunk_id` carries the production `_id` of each chunk, which is what the
ground truth from step 02 refers to. Without it, retrieval results could not
be scored against that ground truth at all.

Writes are idempotent per collection: a collection already holding the full
chunk count is skipped unless --force is given, so an interrupted run can be
resumed without duplicating or double-billing.

Usage:
    python3 eval/scripts/06_populate_eval_collections.py                 # dry run
    python3 eval/scripts/06_populate_eval_collections.py --apply
    python3 eval/scripts/06_populate_eval_collections.py --apply --only voyage_4_large
"""

import argparse
import json
import math
import os
import re
import sys
import time

import dotenv
from pymongo import MongoClient

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv.load_dotenv(os.path.join(REPO, ".env"))

EVAL_DB = "eval"
PROD_DB = os.getenv("PANKB_LLM_DATABASE", "pankb_llm")
PROD_COLLECTION = "pankb_vector_store"

EMBED_BATCH = 96     # texts per embedding API call
INSERT_BATCH = 500   # documents per insert_many


def normalize_doi(d):
    d = (d or "").lower().strip()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    return d.rstrip(".").strip()


def l2_norm(v):
    return math.sqrt(sum(x * x for x in v))


# --- providers -------------------------------------------------------------

def embed_voyage(cfg, texts):
    import voyageai
    client = voyageai.Client()
    kwargs = {"texts": texts, "model": cfg["model"], "input_type": "document"}
    if cfg.get("output_dimension"):
        kwargs["output_dimension"] = cfg["output_dimension"]
    return client.embed(**kwargs).embeddings


def embed_openai(cfg, texts):
    from openai import OpenAI
    client = OpenAI()
    kwargs = {"input": texts, "model": cfg["model"]}
    if cfg.get("dimensions"):
        kwargs["dimensions"] = cfg["dimensions"]
    return [d.embedding for d in client.embeddings.create(**kwargs).data]


def embed_cohere(cfg, texts):
    import cohere
    client = cohere.ClientV2()
    r = client.embed(texts=texts, model=cfg["model"], input_type="search_document",
                     embedding_types=["float"])
    return r.embeddings.float_


EMBEDDERS = {"voyage": embed_voyage, "openai": embed_openai, "cohere": embed_cohere}

# Per-model call arguments. Mirrors step 04 so probe and ingest cannot drift:
# text-embedding-3-large MUST pass dimensions (native 3072 > the 2000 cap), and
# voyage-4-large must never be set to 2048 for the same reason.
MODEL_ARGS = {
    "voyage_large_2_instruct": {},
    "voyage_4_large": {"output_dimension": 1024},
    "openai_3_large_1024": {"dimensions": 1024},
    "cohere_embed_v4": {},
}


def embed_with_retry(cfg, texts, attempts=5):
    """Retry with exponential backoff; rate limits are expected at this volume."""
    for i in range(attempts):
        try:
            return EMBEDDERS[cfg["provider"]](cfg, texts)
        except Exception as e:
            if i == attempts - 1:
                raise
            wait = 2 ** i
            print(f"\n    {type(e).__name__}: {e} — retry in {wait}s "
                  f"({i + 1}/{attempts - 1})", flush=True)
            time.sleep(wait)


def load_chunks(client, dois):
    """Chunk text from production, read-only, ordered for a stable run."""
    col = client[PROD_DB][PROD_COLLECTION]
    chunks = []
    for d in col.find({"source": {"$in": dois}},
                      {"textContent": 1, "source": 1, "title": 1}):
        text = d.get("textContent", "")
        if not text.strip():
            continue
        chunks.append({
            "chunk_id": str(d["_id"]),      # ties results back to the ground truth
            "textContent": text,
            "source": d.get("source", ""),
            "title": d.get("title", ""),
        })
    chunks.sort(key=lambda c: c["chunk_id"])
    return chunks


def populate(db, cfg, chunks, force):
    name = cfg["collection"]
    col = db[name]
    existing = col.estimated_document_count()

    if existing and not force:
        if existing >= len(chunks):
            print(f"  SKIP {name}: already holds {existing:,} documents "
                  f"(use --force to rebuild)")
            return None
        print(f"  {name}: holds {existing:,}/{len(chunks):,} — resuming")
    elif existing and force:
        print(f"  {name}: clearing {existing:,} existing documents")
        col.delete_many({})
        existing = 0

    done = set()
    if existing:
        done = {d["chunk_id"] for d in col.find({}, {"chunk_id": 1})}
    todo = [c for c in chunks if c["chunk_id"] not in done]
    if not todo:
        print(f"  SKIP {name}: nothing left to do")
        return None

    call_cfg = {**cfg, **MODEL_ARGS.get(name, {})}
    print(f"  {name}: embedding {len(todo):,} chunks with {cfg['model']} ...")

    t0 = time.time()
    pending, inserted, norm_lo, norm_hi = [], 0, float("inf"), 0.0
    for i in range(0, len(todo), EMBED_BATCH):
        batch = todo[i:i + EMBED_BATCH]
        vectors = embed_with_retry(call_cfg, [c["textContent"] for c in batch])

        if len(vectors) != len(batch):
            sys.exit(f"ERROR: {name}: asked for {len(batch)} vectors, "
                     f"got {len(vectors)}")
        if len(vectors[0]) != cfg["dimensions"]:
            sys.exit(f"ERROR: {name}: expected {cfg['dimensions']} dims, "
                     f"got {len(vectors[0])} — index would reject these")

        for c, v in zip(batch, vectors):
            n = l2_norm(v)
            norm_lo, norm_hi = min(norm_lo, n), max(norm_hi, n)
            pending.append({
                "chunk_id": c["chunk_id"],
                "textContent": c["textContent"],
                "vectorContent": v,
                "source": c["source"],
                "title": c["title"],
            })

        while len(pending) >= INSERT_BATCH:
            col.insert_many(pending[:INSERT_BATCH])
            inserted += INSERT_BATCH
            pending = pending[INSERT_BATCH:]

        pct = 100 * (i + len(batch)) / len(todo)
        print(f"\r    {pct:5.1f}%  embedded {i + len(batch):,}/{len(todo):,}"
              f"  inserted {inserted:,}", end="", flush=True)

    if pending:
        col.insert_many(pending)
        inserted += len(pending)

    elapsed = time.time() - t0
    total = col.estimated_document_count()
    print(f"\r    done: inserted {inserted:,} | collection holds {total:,} "
          f"| {elapsed / 60:.1f} min" + " " * 20)
    return {
        "collection": name, "model": cfg["model"], "provider": cfg["provider"],
        "dimensions": cfg["dimensions"], "inserted": inserted,
        "total_documents": total, "seconds": round(elapsed, 1),
        "vector_norm_min": round(norm_lo, 6), "vector_norm_max": round(norm_hi, 6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    ap.add_argument("--force", action="store_true", help="clear and rebuild collections")
    ap.add_argument("--only", help="populate a single collection by name")
    args = ap.parse_args()

    probe_path = os.path.join(REPO, "eval", "results", "embedder_probe.json")
    if not os.path.exists(probe_path):
        sys.exit("ERROR: no probe results. Run 04_probe_embedders.py first.")
    models = [r for r in json.load(open(probe_path, encoding="utf-8"))["results"]
              if r.get("ok") and r.get("within_cosmos_limit")]
    if args.only:
        models = [m for m in models if m["collection"] == args.only]
        if not models:
            sys.exit(f"ERROR: no probed model with collection '{args.only}'")

    manifest = json.load(open(os.path.join(REPO, "eval", "corpus",
                                           "subset_manifest.json"), encoding="utf-8"))
    dois = [m["doi"] for m in manifest]
    roles = {normalize_doi(m["doi"]): m["role"] for m in manifest}

    conn = os.getenv("MONGODB_CONN_STRING")
    if not conn:
        sys.exit("ERROR: MONGODB_CONN_STRING not set (expected in .env)")
    client = MongoClient(conn, serverSelectionTimeoutMS=30000)

    print(f"reading chunks for {len(dois)} subset papers from "
          f"{PROD_DB}.{PROD_COLLECTION} (read-only) ...")
    chunks = load_chunks(client, dois)
    n_pos = sum(1 for c in chunks if roles.get(normalize_doi(c["source"])) == "positive")
    approx_tokens = sum(len(c["textContent"]) for c in chunks) / 4
    print(f"  {len(chunks):,} chunks  (positives {n_pos:,} | noise "
          f"{len(chunks) - n_pos:,})")
    print(f"  ~{approx_tokens:,.0f} tokens per model, "
          f"~{approx_tokens * len(models):,.0f} across {len(models)} models\n")

    if not args.apply:
        for m in models:
            col = client[EVAL_DB][m["collection"]]
            print(f"  would populate {m['collection']:<26} dim={m['dimensions']:<5} "
                  f"model={m['model']}  (currently {col.estimated_document_count():,} docs)")
        print("\nDry run — nothing written. Re-run with --apply.")
        return

    db = client[EVAL_DB]
    reports = []
    for m in models:
        r = populate(db, m, chunks, args.force)
        if r:
            reports.append(r)

    if reports:
        path = os.path.join(REPO, "eval", "results", "populate_report.json")
        json.dump({"n_chunks": len(chunks), "n_positive_chunks": n_pos,
                   "runs": reports}, open(path, "w"), indent=1)
        print(f"\nwrote {path}")

    print("\n=== final state ===")
    for m in models:
        print(f"  {m['collection']:<26} "
              f"{client[EVAL_DB][m['collection']].estimated_document_count():,} docs")


if __name__ == "__main__":
    main()
