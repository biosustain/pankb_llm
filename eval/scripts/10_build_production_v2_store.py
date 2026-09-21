#!/usr/bin/env python3
"""
Step 10 — Build a full-corpus vector store with the upgraded embedding model.

Re-embeds every chunk of the production store with `voyage-4-large` into a NEW
collection, leaving the live one untouched. This is the migration the phase-1
results argue for: +0.155 recall@30 at the same 1024 dimensions, the same
price, and the same index definition, so nothing but the vectors changes.

WHY RE-EMBED RATHER THAN RE-CHUNK
---------------------------------
The production store keeps `textContent` alongside each vector, so the chunk
text is already there. Re-embedding it directly means chunk boundaries stay
byte-identical to what is live today and the embedding model is the only thing
that changes -- which is what makes the phase-1 comparison transferable to
this store. Re-splitting the papers would silently introduce a second variable.

SAFETY
------
- Writes only to the new collection; the live `pankb_vector_store` is read-only
  here and remains a zero-cost rollback.
- Dry run by default.
- Resumable: already-embedded chunks are skipped, so an interrupted run does
  not duplicate documents or pay twice.
- The HNSW index is created BEFORE insertion, as production experience
  requires: indexing a populated collection times out unavoidably on this
  cluster tier.
- Each batch is checked against the index width before insertion, so a
  dimension mismatch fails loudly rather than at insert time.

Documents keep `source_id`, the `_id` of the chunk they came from, so any
result in the new store can be traced back to its production original.

Usage:
    python3 eval/scripts/10_build_production_v2_store.py
    python3 eval/scripts/10_build_production_v2_store.py --apply
    python3 eval/scripts/10_build_production_v2_store.py --apply --limit 500
"""

import argparse
import json
import math
import os
import sys
import time

import dotenv
from pymongo import MongoClient

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv.load_dotenv(os.path.join(REPO, ".env"))

PROD_DB = os.getenv("PANKB_LLM_DATABASE", "pankb_llm")
SRC_COLLECTION = "pankb_vector_store"
DST_COLLECTION = "pankb_vector_store_v2"

EMBED_MODEL = "voyage-4-large"
EMBED_DIMS = 1024          # never 2048: Cosmos DB caps vectors at 2000
MAX_DIMS = 2000

# Mirror the live index exactly, so the embedding model is the only change.
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 100
HNSW_SIMILARITY = "L2"

EMBED_BATCH = 96
INSERT_BATCH = 500


def l2_norm(v):
    return math.sqrt(sum(x * x for x in v))


def embed(texts, attempts=5):
    """Voyage-4 takes input_type rather than the instruction prefixes the
    legacy voyage-large-2-instruct model expected."""
    import voyageai
    client = voyageai.Client()
    for i in range(attempts):
        try:
            return client.embed(texts=texts, model=EMBED_MODEL,
                                input_type="document",
                                output_dimension=EMBED_DIMS).embeddings
        except Exception as e:
            if i == attempts - 1:
                raise
            wait = 2 ** i
            print(f"\n  {type(e).__name__}: {e} — retry in {wait}s", flush=True)
            time.sleep(wait)


def ensure_index(db, name):
    existing = {i["name"] for i in db[name].list_indexes()} if name in db.list_collection_names() else set()
    idx_name = f"{name}_hnsw_index"
    if idx_name in existing:
        print(f"  index {idx_name} already present")
        return
    db.command({
        "createIndexes": name,
        "indexes": [{
            "name": idx_name,
            "key": {"vectorContent": "cosmosSearch"},
            "cosmosSearchOptions": {
                "kind": "vector-hnsw", "m": HNSW_M,
                "efConstruction": HNSW_EF_CONSTRUCTION,
                "similarity": HNSW_SIMILARITY, "dimensions": EMBED_DIMS,
            },
        }],
    })
    db[name].create_index("source")
    db[name].create_index("source_id", unique=True)
    print(f"  created {idx_name} (dim={EMBED_DIMS}, {HNSW_SIMILARITY})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    ap.add_argument("--limit", type=int, help="only process N chunks (for a trial run)")
    ap.add_argument("--dst", default=DST_COLLECTION)
    args = ap.parse_args()

    if EMBED_DIMS > MAX_DIMS:
        sys.exit(f"ERROR: {EMBED_DIMS} dims exceeds the Cosmos DB limit of {MAX_DIMS}")

    conn = os.getenv("MONGODB_CONN_STRING")
    if not conn:
        sys.exit("ERROR: MONGODB_CONN_STRING not set (expected in .env)")
    client = MongoClient(conn, serverSelectionTimeoutMS=30000)
    db = client[PROD_DB]
    src, dst = db[SRC_COLLECTION], db[args.dst]

    n_src = src.count_documents({})
    n_dst = dst.count_documents({}) if args.dst in db.list_collection_names() else 0

    print(f"source      : {PROD_DB}.{SRC_COLLECTION}  ({n_src:,} chunks, read-only)")
    print(f"destination : {PROD_DB}.{args.dst}  ({n_dst:,} already present)")
    print(f"model       : {EMBED_MODEL} @ {EMBED_DIMS} dims")
    print(f"index       : HNSW m={HNSW_M} ef={HNSW_EF_CONSTRUCTION} "
          f"{HNSW_SIMILARITY}  (mirrors production)\n")

    if not args.apply:
        print("Dry run — nothing written. Re-run with --apply.")
        return

    ensure_index(db, args.dst)

    done = {d["source_id"] for d in dst.find({}, {"source_id": 1})} if n_dst else set()
    cursor = src.find({}, {"textContent": 1, "source": 1, "title": 1}, no_cursor_timeout=True)

    pending, todo, inserted, skipped = [], [], 0, 0
    norm_lo, norm_hi = float("inf"), 0.0
    t0 = time.time()

    def flush_embeddings():
        nonlocal pending, todo, inserted, norm_lo, norm_hi
        if not todo:
            return
        vectors = embed([c["textContent"] for c in todo])
        if len(vectors) != len(todo):
            sys.exit(f"ERROR: asked for {len(todo)} vectors, got {len(vectors)}")
        if len(vectors[0]) != EMBED_DIMS:
            sys.exit(f"ERROR: model returned {len(vectors[0])} dims, index expects "
                     f"{EMBED_DIMS} — insertion would be rejected")
        for c, v in zip(todo, vectors):
            n = l2_norm(v)
            norm_lo, norm_hi = min(norm_lo, n), max(norm_hi, n)
            pending.append({
                "source_id": c["source_id"], "textContent": c["textContent"],
                "vectorContent": v, "source": c["source"], "title": c["title"],
            })
        todo = []
        while len(pending) >= INSERT_BATCH:
            dst.insert_many(pending[:INSERT_BATCH])
            inserted += INSERT_BATCH
            pending = pending[INSERT_BATCH:]

    try:
        for i, d in enumerate(cursor):
            if args.limit and (inserted + len(pending) + len(todo)) >= args.limit:
                break
            sid = str(d["_id"])
            if sid in done:
                skipped += 1
                continue
            text = d.get("textContent", "")
            if not text.strip():
                skipped += 1
                continue
            todo.append({"source_id": sid, "textContent": text,
                         "source": d.get("source", ""), "title": d.get("title", "")})
            if len(todo) >= EMBED_BATCH:
                flush_embeddings()
                el = time.time() - t0
                rate = inserted / el if el else 0
                print(f"\r  inserted {inserted:,}  skipped {skipped:,}  "
                      f"{rate * 60:.0f}/min  elapsed {el / 60:.1f}m", end="", flush=True)
        flush_embeddings()
        if pending:
            dst.insert_many(pending)
            inserted += len(pending)
    finally:
        cursor.close()

    total = dst.count_documents({})
    print(f"\r  done: inserted {inserted:,} | skipped {skipped:,} | "
          f"collection holds {total:,} | {(time.time() - t0) / 60:.1f} min" + " " * 15)

    report = {
        "source": f"{PROD_DB}.{SRC_COLLECTION}", "destination": f"{PROD_DB}.{args.dst}",
        "model": EMBED_MODEL, "dimensions": EMBED_DIMS,
        "source_chunks": n_src, "inserted": inserted, "total_documents": total,
        "vector_norm_min": round(norm_lo, 6) if norm_lo != float("inf") else None,
        "vector_norm_max": round(norm_hi, 6),
        "index": {"m": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION,
                  "similarity": HNSW_SIMILARITY},
    }
    path = os.path.join(REPO, "eval", "results", "production_v2_build.json")
    json.dump(report, open(path, "w"), indent=1)
    print(f"\nwrote {path}")

    if total < n_src:
        print(f"\nNOTE: {n_src - total:,} chunks are not in the new collection. "
              f"Re-run to resume.")


if __name__ == "__main__":
    main()
