#!/usr/bin/env python3
"""
Step 04 — Probe the candidate embedding models before building anything.

Runs three short texts through each candidate and reports what actually comes
back. Nothing is written to the database. This has to happen first because:

  1. The HNSW index must declare `dimensions` at CREATION time, and Cosmos DB
     rejects inserts that don't match. So the true output width has to be
     measured, not assumed.
  2. The production index uses `similarity: "L2"`, not cosine. L2 and cosine
     rank identically ONLY for normalized vectors -- an assumption that holds
     for the current model but is written down nowhere. A model returning
     non-unit vectors would silently produce a wrong ranking, and every recall
     number computed from it would be wrong in a way no sanity check catches.
     Truncated vectors are the real risk: shortening a unit vector generally
     does not leave it unit-length.
  3. A missing key or an unauthorized model should surface in seconds, not
     halfway through embedding 15k chunks.

Cosmos DB for MongoDB vCore caps vectors at 2000 dimensions, so any model
exceeding that must be reduced at call time (OpenAI `dimensions`, Voyage
`output_dimension`).

Usage:
    python3 eval/scripts/04_probe_embedders.py
"""

import json
import math
import os
import sys

import dotenv

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv.load_dotenv(os.path.join(REPO, ".env"))

MAX_DIMS = 2000  # Azure Cosmos DB for MongoDB vCore hard limit

PROBE_TEXTS = [
    "Pangenome analysis reveals the genetic basis for taxonomic classification.",
    "Lactate dehydrogenase is a crucial enzyme in the bacterial lactic acid "
    "synthesis pathway.",
    "The core genome comprises genes present in more than 99% of strains.",
]

# `collection` is the MongoDB collection name: underscores only, since hyphens
# force quoting in shells and some tooling.
CANDIDATES = [
    {
        "key": "voyage_large_2_instruct",
        "provider": "voyage",
        "model": "voyage-large-2-instruct",
        "collection": "voyage_large_2_instruct",
        "note": "production baseline (now legacy upstream)",
    },
    {
        "key": "voyage_4_large",
        "provider": "voyage",
        "model": "voyage-4-large",
        "collection": "voyage_4_large",
        "output_dimension": 1024,  # never 2048: exceeds the Cosmos cap
        "note": "vendor-recommended successor to the baseline",
    },
    {
        "key": "openai_3_large_1024",
        "provider": "openai",
        "model": "text-embedding-3-large",
        "collection": "openai_3_large_1024",
        "dimensions": 1024,  # REQUIRED: native 3072 exceeds the Cosmos cap
        "note": "cross-vendor comparison; truncated from 3072",
    },
    {
        "key": "cohere_embed_v4",
        "provider": "cohere",
        "model": "embed-v4.0",
        "collection": "cohere_embed_v4",
        "note": "cross-vendor comparison; 1536 native",
    },
]


def l2_norm(v):
    return math.sqrt(sum(x * x for x in v))


def embed_voyage(cfg, texts, input_type):
    import voyageai
    client = voyageai.Client()
    kwargs = {"texts": texts, "model": cfg["model"], "input_type": input_type}
    if "output_dimension" in cfg:
        kwargs["output_dimension"] = cfg["output_dimension"]
    return client.embed(**kwargs).embeddings


def embed_openai(cfg, texts, input_type):
    from openai import OpenAI
    client = OpenAI()
    kwargs = {"input": texts, "model": cfg["model"]}
    if "dimensions" in cfg:
        kwargs["dimensions"] = cfg["dimensions"]
    return [d.embedding for d in client.embeddings.create(**kwargs).data]


def embed_cohere(cfg, texts, input_type):
    import cohere
    client = cohere.ClientV2()
    r = client.embed(
        texts=texts,
        model=cfg["model"],
        input_type="search_query" if input_type == "query" else "search_document",
        embedding_types=["float"],
    )
    return r.embeddings.float_


EMBEDDERS = {"voyage": embed_voyage, "openai": embed_openai, "cohere": embed_cohere}

REQUIRED_KEYS = {
    "voyage": "VOYAGE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "cohere": "COHERE_API_KEY",
}


def probe(cfg):
    out = {
        "key": cfg["key"], "provider": cfg["provider"], "model": cfg["model"],
        "collection": cfg["collection"], "note": cfg.get("note", ""),
        "ok": False, "error": None,
    }

    env_key = REQUIRED_KEYS[cfg["provider"]]
    if not os.getenv(env_key):
        out["error"] = f"{env_key} not set in .env"
        return out

    try:
        doc_vecs = EMBEDDERS[cfg["provider"]](cfg, PROBE_TEXTS, "document")
        # Queries are embedded separately at retrieval time; confirm the query
        # path works too and returns the same width.
        qry_vecs = EMBEDDERS[cfg["provider"]](cfg, [PROBE_TEXTS[0]], "query")
    except Exception as e:  # surface the real reason, don't mask it
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    dims = len(doc_vecs[0])
    norms = [l2_norm(v) for v in doc_vecs]
    out.update({
        "ok": True,
        "dimensions": dims,
        "query_dimensions": len(qry_vecs[0]),
        "norm_min": round(min(norms), 6),
        "norm_max": round(max(norms), 6),
        "normalized": all(abs(n - 1.0) < 1e-3 for n in norms),
        "within_cosmos_limit": dims <= MAX_DIMS,
        "dim_matches_query": dims == len(qry_vecs[0]),
    })
    return out


def main():
    print(f"probing {len(CANDIDATES)} embedding models "
          f"({len(PROBE_TEXTS)} texts each, nothing written to the database)\n")

    results = []
    for cfg in CANDIDATES:
        print(f"  {cfg['key']:<26} ... ", end="", flush=True)
        r = probe(cfg)
        results.append(r)
        if r["ok"]:
            flags = []
            if not r["within_cosmos_limit"]:
                flags.append(f"EXCEEDS {MAX_DIMS}-DIM LIMIT")
            if not r["normalized"]:
                flags.append("NOT NORMALIZED")
            if not r["dim_matches_query"]:
                flags.append("QUERY/DOC DIM MISMATCH")
            status = "OK" if not flags else "!! " + "; ".join(flags)
            print(f"dim={r['dimensions']:<5} norm=[{r['norm_min']}, "
                  f"{r['norm_max']}]  {status}")
        else:
            print(f"FAILED — {r['error']}")

    usable = [r for r in results
              if r["ok"] and r["within_cosmos_limit"] and r["dim_matches_query"]]
    unnormalized = [r for r in usable if not r["normalized"]]

    print("\n=== summary ===")
    print(f"  usable: {len(usable)}/{len(CANDIDATES)}")
    for r in usable:
        print(f"    {r['key']:<26} dim={r['dimensions']:<5} "
              f"normalized={r['normalized']}  collection={r['collection']}")
    for r in results:
        if not r["ok"]:
            print(f"    UNUSABLE {r['key']}: {r['error']}")

    if unnormalized:
        print("\n  WARNING: these return non-unit vectors, so L2 and cosine do "
              "NOT rank identically:")
        for r in unnormalized:
            print(f"    {r['key']}  norms [{r['norm_min']}, {r['norm_max']}]")
        print("  They must be normalized before insertion, or their recall "
              "numbers will be wrong.")

    out_path = os.path.join(REPO, "eval", "results", "embedder_probe.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({"max_dims": MAX_DIMS, "probe_texts": PROBE_TEXTS,
               "results": results}, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")

    if not usable:
        sys.exit("ERROR: no usable embedding model; cannot proceed to build.")


if __name__ == "__main__":
    main()
