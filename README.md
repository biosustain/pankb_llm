# PanKB LLM (DEV)

## Overview

A GenAI Assistant based on Langchain + Streamlit + Azure Cosmos DB for MongoDB (vCore) + Docker.

Authors:
- Data preprocessing, LLM, DEV vector DB creation (Chroma), retriever, Streamlit web app: Binhuan Sun, binsun@biosustain.dtu.dk 
- Changing the DEV vector DB (Chroma) to the PROD vector DB instance (Azure Cosmos DB for MongoDB (vCore)) and adjusting the choice of embeddings to the Cosmos DB limitations, the PROD DB index creation, dockerization, integration of the streamlit app with the Django framework templates, the github repo maintenance: Pashkova Liubov, liupa@dtu.dk

## Server Configuration
Tested on Linux Ubuntu 20.04 (may need tweaks for other systems).

Min hardware requirements solely for the AI Assistant App deployment and Vector database creation (excl. the PanKB DB, ETL and AI Assistant app):
- 8GB RAM
- 8GB disk space
- 4 CPU cores (e.g., for PyCharm Remote IDE development)

System requirements:
- Docker & Docker Compose
- Git

## Important considerations & limitations

The DB population process can take up to 90-150 minutes (depends on the DEV server configuration). The MongoDB storage size the populated collection is ~ 1.0 GiB, incl. the indexes.

Please note the following limitations and considerations:
- If we use an Azure Cosmos DB for MongoDB instance as the vector DB, we can try only embeddings with dimensionalities <= 2000 because for Azure Cosmos DB for MongoDB the maximum number of supported dimensions is 2000. Maybe it is even for the better, large embeddings are more expensive and not always provide a significant increase in performance. Examples: https://platform.openai.com/docs/guides/embeddings 
- We have to create the similarity index. The dimensionality of this index must be the same as the dimensionality of the embeddings.
- The Azure M30 tier (Azure Cosmos DB for MongoDB) supports only the <i>vector-ivf</i> index type. To create the <i>vector-hnsw</i> index, we have to upgrade to the M40 tier (it costs twice more than M30 if we do not select the "High Availability" option on Azure Portal).
- We have to create the HNSW index before data insertion (although it significantly increases data insertion time). If we do it after, we can not avoid the timeout-type error. The reasons can be the cluster configuration options that we can not change on our side if we use IaaS.

## Scripts execution

Create the .env file in the following format:
```
## Do not put this file under version control!

OPENAI_API_KEY=<insert the API key here without quotes>
COHERE_API_KEY=<insert the API key here without quotes>
TOGETHER_API_KEY=<insert the API key here without quotes>
GOOGLE_API_KEY=<insert the API key here without quotes>
ANTHROPIC_API_KEY=<insert the API key here without quotes>
REPLICATE_API_TOKEN=<insert the API key here without quotes>
VOYAGE_API_KEY=<insert the API key here without quotes>

## MongoDB-PROD (Azure Cosmos DB for MongoDB) Connection String
# Had to multiply maxIdleTimeMS by 10 to handle
# urllib3.exceptions.ProtocolError: 
# ("Connection broken: ConnectionResetError(104, 'Connection reset by peer')", ConnectionResetError(104, 'Connection reset by peer'))
MONGODB_CONN_STRING = "<insert the connection string here with quotes>"
```

The DB population script does not have to be executed in a docker container. It can be done with the following commands:
```
# install all the requirements and dependencies
pip3 install -r requirements.txt

# Run the script with two command line arguments: 
# the name of the folder containing the documents to feed to the LLM 
# and 
# the name of the MongoDB collection that will contain the vector DB
python3 make_vectordb.py ./Paper_all pankb_vector_store
```
The command for building the docker image and recreating the docker container with the Streamlit app inside:
```
docker compose up -d --build --force-recreate
```
The dockerized streamlit app does not have to be executed in <i>tmux</i>. It will always be up and running even after the VM is rebooted (achieved by using the option `restart: always` in the docker compose file).

The status of the docker container can be checked with the following command:
```
docker ps
```
The command should produce approx. the following output among others in case of successful deployment:
```
CONTAINER ID   IMAGE                COMMAND                  CREATED          STATUS          PORTS                                           NAMES
54d89d7c4fad   pankb_llm:latest     "streamlit run strea…"   10 minutes ago   Up 10 minutes   0.0.0.0:8501->8501/tcp, :::8501->8501/tcp       pankb-llm
```