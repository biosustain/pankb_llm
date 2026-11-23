#!/usr/bin/env python3

"""
Usage:
    python backup_vectordb.py  (for a dry run)
    python backup_vectordb.py --apply
"""
import argparse
import datetime
import os
import shutil
import subprocess
import sys

import dotenv

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv.load_dotenv(os.path.join(REPO, ".env"))

DB = os.getenv("PANKB_LLM_DATABASE", "pankb_llm")
COLLECTIONS = ["pankb_vector_store_v2"] # change if necessary
CONTAINER_URL = "https://pankbpipeline.blob.core.windows.net/vectordb"

OUT_DIR = os.getenv("PANKB_LLM_BACKUP_DIR", "/data/pankb_llm_backups")

MONGO_URI = os.getenv("MONGODB_CONN_STRING")
AZURE_BLOB_SAS = os.getenv("AZURE_BLOB_SAS")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="dump and upload")
    args = ap.parse_args()

    mongodump = shutil.which("mongodump")

    date = datetime.date.today().isoformat()
    day_dir = os.path.join(OUT_DIR, date)
    print(f"database    : {DB}")
    print(f"collections : {COLLECTIONS}")
    print(f"local dir   : {day_dir}")
    print(f"destination : {CONTAINER_URL}/{date}/<collection>.archive.gz")
    
    if not args.apply:
        print("\nDry run — nothing dumped or uploaded. Re-run with --apply.")
        return

    from azure.storage.blob import ContainerClient, StandardBlobTier

    container = ContainerClient.from_container_url(f"{CONTAINER_URL}?{AZURE_BLOB_SAS}")

    os.makedirs(day_dir, exist_ok=True)
    for coll in COLLECTIONS:
        path = os.path.join(day_dir, f"{coll}.archive.gz")
        print(f"\n[{coll}] dumping ...")
        r = subprocess.run([mongodump, "--uri", MONGO_URI, "--db", DB,
                            "--collection", coll, f"--archive={path}", "--gzip"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"ERROR: mongodump failed for {coll}: "
                     f"{r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ''}")
        size_mb = os.path.getsize(path) / 1e6
        print(f"[{coll}] {size_mb:,.0f} MB -> uploading ...")

        blob = container.get_blob_client(f"{date}/{coll}.archive.gz")
        with open(path, "rb") as f:
            # overwrite=False: an existing backup for today is never replaced.
            blob.upload_blob(f, overwrite=False, max_concurrency=4,
                             standard_blob_tier=StandardBlobTier.COLD)
        print(f"[{coll}] uploaded to {CONTAINER_URL}/{date}/{coll}.archive.gz")
        
    shutil.rmtree(day_dir)
    print("\ndone")


if __name__ == "__main__":
    main()
