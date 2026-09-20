#!/usr/bin/env python3
"""
Step 05 — Create the `eval` database and one collection per embedding model.

This is the first script that WRITES to the Cosmos DB cluster. It touches only
the `eval` database; the production database is never opened.

Collections are created empty but WITH their HNSW index already in place. That
order is deliberate and inherited from production experience (documented in the
repo README): creating the index after populating the collection reliably times
out on this cluster tier, and the timeout cannot be worked around from the
client side. Building the index first slows insertion but completes.

`dimensions` is read from the probe results (step 04) rather than assumed — the
index declares a fixed width and Cosmos rejects any vector that disagrees with
it, so the value has to be the measured one.

Index parameters (m=16, efConstruction=100, similarity=L2) mirror production
exactly. This experiment compares embedding models, so every other variable is
held at the production setting — including the L2 metric, even though cosine
would be the more usual choice for normalized vectors. Step 04 confirmed all
four candidates return unit-norm vectors, which is what makes L2 and cosine
rank identically; without that check this index choice would be unsafe.

Usage:
    python3 eval/scripts/05_create_eval_collections.py            # dry run
    python3 eval/scripts/05_create_eval_collections.py --apply    # create
    python3 eval/scripts/05_create_eval_collections.py --apply --drop
"""

import argparse
import json
import os
import sys

import dotenv
from pymongo import MongoClient

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv.load_dotenv(os.path.join(REPO, ".env"))

EVAL_DB = "eval"
PROD_DB = os.getenv("PANKB_LLM_DATABASE", "pankb_llm")

# Mirror production exactly (see make_vectordb_native.py).
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 100
HNSW_SIMILARITY = "L2"
MAX_DIMS = 2000


def load_probe():
    path = os.path.join(REPO, "eval", "results", "embedder_probe.json")
    if not os.path.exists(path):
        sys.exit("ERROR: no probe results. Run 04_probe_embedders.py first.")
    probe = json.load(open(path, encoding="utf-8"))
    usable = [r for r in probe["results"]
              if r.get("ok") and r.get("within_cosmos_limit")
              and r.get("dim_matches_query")]
    if not usable:
        sys.exit("ERROR: probe found no usable embedding model.")
    return usable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually create (default is a dry run)")
    ap.add_argument("--drop", action="store_true",
                    help="drop an existing collection before recreating it")
    args = ap.parse_args()

    models = load_probe()

    print(f"target database : {EVAL_DB}  (production `{PROD_DB}` is not touched)")
    print(f"index           : HNSW m={HNSW_M} efConstruction={HNSW_EF_CONSTRUCTION} "
          f"similarity={HNSW_SIMILARITY}")
    print(f"mode            : {'APPLY' if args.apply else 'DRY RUN'}"
          f"{' + DROP' if args.drop else ''}\n")

    for m in models:
        warn = "" if m.get("normalized") else "   <-- NOT NORMALIZED: L2 ranking unsafe"
        print(f"  {m['collection']:<26} dim={m['dimensions']:<5} "
              f"model={m['model']}{warn}")

    if any(not m.get("normalized") for m in models):
        print("\nWARNING: a model returns non-unit vectors. With an L2 index its "
              "ranking will differ from cosine and its recall numbers will be "
              "wrong. Normalize before inserting, or exclude the model.")

    if not args.apply:
        print("\nDry run — nothing created. Re-run with --apply to create.")
        return

    conn = os.getenv("MONGODB_CONN_STRING")
    if not conn:
        sys.exit("ERROR: MONGODB_CONN_STRING not set (expected in .env)")
    client = MongoClient(conn, serverSelectionTimeoutMS=30000)
    db = client[EVAL_DB]

    existing = set(db.list_collection_names())
    print(f"\nexisting collections in `{EVAL_DB}`: "
          f"{sorted(existing) if existing else '(none)'}\n")

    created = []
    for m in models:
        name, dims = m["collection"], m["dimensions"]
        if dims > MAX_DIMS:  # belt and braces; the probe already filtered these
            print(f"  SKIP {name}: {dims} dims exceeds the {MAX_DIMS} limit")
            continue

        if name in existing:
            if args.drop:
                print(f"  dropping existing {name} ...")
                db[name].drop()
            else:
                print(f"  SKIP {name}: already exists (use --drop to recreate)")
                continue

        db.command({
            "createIndexes": name,
            "indexes": [{
                "name": f"{name}_hnsw_index",
                "key": {"vectorContent": "cosmosSearch"},
                "cosmosSearchOptions": {
                    "kind": "vector-hnsw",
                    "m": HNSW_M,
                    "efConstruction": HNSW_EF_CONSTRUCTION,
                    "similarity": HNSW_SIMILARITY,
                    "dimensions": dims,
                },
            }],
        })
        # Metadata travels with the data: a collection whose provenance is not
        # recorded cannot be interpreted later.
        db[name].create_index("source")
        print(f"  created {name} (dim={dims}) + HNSW index")
        created.append({
            "collection": name, "model": m["model"], "provider": m["provider"],
            "dimensions": dims, "normalized": m.get("normalized"),
            "hnsw": {"m": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION,
                     "similarity": HNSW_SIMILARITY},
        })

    if created:
        path = os.path.join(REPO, "eval", "results", "eval_collections.json")
        json.dump({"database": EVAL_DB, "collections": created},
                  open(path, "w"), indent=1)
        print(f"\nwrote {path}")

    print(f"\ncollections now in `{EVAL_DB}`: {sorted(db.list_collection_names())}")


if __name__ == "__main__":
    main()
