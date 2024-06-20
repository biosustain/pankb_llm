# README

## Overview

A new GenAI chat application that uses an Azure Cosmos DB for MongoDB instance instead of Chroma.

Authors: Binhuan Sun (binsun@biosustain.dtu.dk), Pashkova Liubov (liupa@dtu.dk)

## Changes

The changes were made to the following files: 
- `.env` file (added a MongoDB connecting string as a new environment variable);
- `make_vectordb.py`;
- `streamlit_app.py`;
- the streamlit app is dockerized.

Changelog:
- the embedding model has to be different (see below);
- the index size is different also;
- unnecessary command lines arguments are removed;
- added execution time calculation for the vector db population script.

## Important considerations & limitations

The DB population process took 27 minutes. The MongoDB storage size the populated collection took is ~ 1.3 GiB, incl. the indexes.

Please note the following limitations and considerations:
- If we use an Azure Cosmos DB for MongoDB as the vector db instead of Chroma, we can try only one of the following embeddings: 'text-embedding-3-small' (dimensionality = 1536) or 'text-embedding-ada-002' (dimensionality = 1536), because for Azure Cosmos DB for MongoDB the maximum number of supported dimensions is 2000. Maybe it is even for the better, because OpenAI themselves do not recommend using large embeddings. Additionally, large embeddings are more expensive. Source: https://platform.openai.com/docs/guides/embeddings
- Now we have to create the similarity index and the dimensionality of this index must match the dimensionality of the embeddings (in our case 1536).
- The CPU (M30) on a server, where we have our Azure Cosmos DB for MongoDB instance, supports only the <i>vector-ivf</i> index type. To create the <i>vector-hnsw</i> index, we need to upgrade to the M40 tier (it will cost us 780.42 USD per month instead of 211.36 that we pay for M30 now).

## Scripts execution
The DB population script does not have to be executed in a docker container:
```
python make_vectordb.py ./Paper_all pankb_vector_store
```
(now we don't need the third argument, as we do not persist anything to the RAM; we insert to the MongoDB instance)

and building and rebuilding the container with the streamlit app:
```
docker compose up -d --build
```
The dockerized streamlit app does not have to be executed in tmux. It will always be up and running even after the VM is rebooted (achieved by using the option `restart: always` in the docker compose file).

The status of the docker container cna be checked with the following command:
```
docker ps
```
The command should produce approx. the following output among others:
```
CONTAINER ID   IMAGE                COMMAND                  CREATED          STATUS          PORTS                                           NAMES
54d89d7c4fad   pankb_llm:latest     "streamlit run strea…"   10 minutes ago   Up 10 minutes   0.0.0.0:8501->8501/tcp, :::8501->8501/tcp       pankb-llm
```

## Availability

Currently, the new MongoDB-based streamlit app is up and running on the pakb-dev VM and available by the IP address:
```
http://52.169.153.8:8501/
```
The Chroma-based version is available on the pankb-ale VM and integrated into a separate Django module on the pankb-dev VM: 
```
http://52.169.153.8/ai_assistant/
```
The responses of these two different bots can be compared. If the code on the pankb-ale VM is substituted, the changes will be reflected immediately in the iframe of the Django web application.