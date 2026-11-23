# PanKB LLM (PROD)

## Overview

A GenAI Assistant based on Langchain + Streamlit + Azure Cosmos DB for MongoDB (vCore) + Docker.

Authors:
- Data preprocessing, LLM, DEV vector DB creation (Chroma), retriever, Streamlit web app: Binhuan Sun, binsun@biosustain.dtu.dk 
- Changing the DEV vector DB (Chroma) to the PROD vector DB instance (Azure Cosmos DB for MongoDB (vCore)) and adjusting the choice of embeddings to the Cosmos DB limitations, the PROD DB index creation, dockerization, integration of the streamlit app with the Django framework templates, the github repo maintenance: Pashkova Liubov, liupa@dtu.dk

## Current configuration

| Component | Model / setting |
|---|---|
| Vector store | `pankb_vector_store_v2` (Azure Cosmos DB for MongoDB vCore, 1024-dim, HNSW) |
| Embedder | `voyage-4-large` (Voyage, 1024 dims) |
| Retrieval | top 30 |
| Reranker | `rerank-2.5` (Voyage), top 20, threshold 0.45 |
| Generator | `gpt-6-luna` (OpenAI), `reasoning_effort="none"`, `temperature=0` |

Chosen by the evaluation in [eval/](eval/); see [eval/FINDINGS.md](eval/FINDINGS.md).
Updated 2026-11-22.

## Backups and restore

[backup/backup_vectordb.py](backup/backup_vectordb.py) dumps collections in Azure Cosmos DB's pankb_llm database with `mongodump` and uploads them to the `vectordb` container of
the `pankbpipeline` storage account. Run it in the
`pankb_llm` conda env after any rebuild of the vector store or bulk paper ingestion:

### Setup

Install the dependencies into the conda env after activating it.
    
```
conda activate pankb_llm
conda install -c conda-forge mongo-tools
pip install -r backup/requirements.txt
```

### Backup

```
conda activate pankb_llm
python backup/backup_vectordb.py            # dry run: show the plan
python backup/backup_vectordb.py --apply    # dump and upload
```

Each collection is uploaded as `vectordb/<YYYY-MM-DD>/<collection>.archive.gz`.
Update `AZURE_BLOB_SAS` in `.env` when necessary.     

### Restore

1. Download the archive (the container URL plus the SAS token):

   ```
   conda activate pankb_llm
   set -a; source .env; set +a
   curl -o pankb_vector_store_v2.archive.gz \
     "https://pankbpipeline.blob.core.windows.net/vectordb/<YYYY-MM-DD>/pankb_vector_store_v2.archive.gz?${AZURE_BLOB_SAS}"
   ```

2. Create the target collection with all its indexes first, while it is still empty. The settings below match the live `pankb_vector_store_v2`
   (1024 dims, `m=16`, `efConstruction=100`, `similarity: L2`, plus the `source`
   and unique `source_id` indexes):

   ```
   python -c "
   import os, dotenv; dotenv.load_dotenv('.env')
   from pymongo import MongoClient
   target = 'pankb_vector_store_v2_restored'
   db = MongoClient(os.environ['MONGODB_CONN_STRING'])[os.getenv('PANKB_LLM_DATABASE', 'pankb_llm')]
   db.create_collection(target)
   db.command({'createIndexes': target, 'indexes': [
       {'name': target + '_hnsw_index', 'key': {'vectorContent': 'cosmosSearch'},
        'cosmosSearchOptions': {'kind': 'vector-hnsw', 'm': 16, 'efConstruction': 100,
                                'similarity': 'L2', 'dimensions': 1024}},
       {'name': 'source_1', 'key': {'source': 1}},
       {'name': 'source_id_1', 'key': {'source_id': 1}, 'unique': True}]})
   print([ix['name'] for ix in db[target].list_indexes()])
   "
   ```

3. Restore into that new collection. `--noIndexRestore` skips the index
   definitions stored in the archive, since step 2 already created them:

   ```
   mongorestore --uri "$MONGODB_CONN_STRING" --archive=pankb_vector_store_v2.archive.gz --gzip \
     --nsFrom "ankb_llm.*" --nsTo "pankb_llm.*_restored" \
     --noIndexRestore
   ```

4. Check the document count, then point `collection_name` in `streamlit_app.py`
   at the restored collection.

Restoring into a new collection (`_restored`) never touches the live one, so a
bad archive cannot damage production.

## Scripts execution

Every time when one pushes to the `prod` repo (usually from the DEV server), the changes in the AI Assistant Web Application will be AUTOMATICALLY deployed to the PROD server. The automation (CI/CD) is achieved with the help of Github Actions enabled for the repository. The respective config file is `.github/workflows/deploy-prod-to-azurevm.yml`. In order for the automated deployment to work, you should set up the values of the following secret Github Actions secrets:
```
PANKB_PROD_HOST - the PROD server IP address
PANKB_PROD_SSH_USERNAME - the ssh user name to connect to the PROD server
PANKB_PROD_PRIVATE_SSH_KEY - the ssh key that is used to connect to the PROD server
OPENAI_API_KEY
COHERE_API_KEY
TOGETHER_API_KEY
GOOGLE_API_KEY
ANTHROPIC_API_KEY
REPLICATE_API_TOKEN
VOYAGE_API_KEY
AZURE_BLOB_SAS
PANKB_PROD_MONGODB_CONN_STRING - MongoDB PROD (Azure CosmosDB for MongoDB) Connection String
```
These secrets are encrypted and safely stored on Github in the "Settings - Secrets and Variables - Actions - Repository secrets" section. In this section, you can also add new Github Actions secrets and edit the existing ones. However, in order to change a secret name, you have to remove the existing secret and add the new one instead of the old one.

After the Github Actions deployment job has successfully run, the web-application must be available at [pankb.org/ai_assistant](https://pankb.org/ai_assistant/). 

The status of the docker container can be checked with the following command:
```
docker ps
```
The command should produce approx. the following output in case of the successful deployment:
```
CONTAINER ID   IMAGE                COMMAND                  CREATED          STATUS          PORTS                                           NAMES
54d89d7c4fad   pankb_llm:latest     "streamlit run strea…"   23 seconds ago   Up 12 seconds   0.0.0.0:8501->8501/tcp, :::8501->8501/tcp       pankb-llm
```