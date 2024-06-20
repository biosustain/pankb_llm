# README

## Overview

A GenAI Assistant based on Langchain + Streamlit + Azure Cosmos DB for MongoDB (vCore) + Docker.

Authors:
- Binhuan Sun (binsun@biosustain.dtu.dk): data preprocessing, LLM, DEV vector DB creation (Chroma), retriever, Streamlit web app. 
- Pashkova Liubov (liupa@dtu.dk): Changing the DEV vector DB (Chroma) to the PROD vector DB instance (Azure Cosmos DB for MongoDB (vCore)) and adjusting the choice of embeddings to the Cosmos DB limitations, the PROD DB index creation, dockerization, integration of the streamlit app with the Django framework templates, the Github repo set up.

## Important considerations & limitations

The DB population process took 93 minutes (<i>toedit: goddamn long!!!introduce multithreading to speed up!!!</i>). The MongoDB storage size the populated collection took is ~ 1.0 GiB, incl. the indexes.

Please note the following limitations and considerations:
- If we use an Azure Cosmos DB for MongoDB instance as the vector DB, we can try only embeddings with dimensionalities <= 2000 because for Azure Cosmos DB for MongoDB the maximum number of supported dimensions is 2000. Maybe it is even for the better, large embeddings are more expensive and not always provide a significant increase in performance. Examples: https://platform.openai.com/docs/guides/embeddings 
- We have to create the similarity index. The dimensionality of this index must match the dimensionality of the embeddings.
- The CPU (M30) on a server, where we have our Azure Cosmos DB for MongoDB instance, supports only the <i>vector-ivf</i> index type. To create the <i>vector-hnsw</i> index, we need to upgrade to the M40 tier (it will cost us 780.42 USD per month instead of 211.36 that we pay for M30 now).

## Scripts execution

Create the .env file in the following format:
```
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
The connection string and API keys can be obtained from the authors.

The DB population script does not have to be executed in a docker container. It can be done with the following commands:
```
pip3 install -r requirements.txt

python3 make_vectordb.py ./Paper_all pankb_vector_store
```
The first command above installs all the requirements. The second one runs the script with two command line arguments: the name of the folder containing the documents to feed to the LLM and the name of the MongoDB collection that will contain the vector DB. 

The command for building and rebuilding the docker container with the Streamlit app inside:
```
docker compose up -d --build --force-recreate
```
The dockerized streamlit app does not have to be executed in <i>tmux></i>. It will always be up and running even after the VM is rebooted (achieved by using the option `restart: always` in the docker compose file).

The status of the docker container can be checked with the following command:
```
docker ps
```
The command should produce approx. the following output among others:
```
CONTAINER ID   IMAGE                COMMAND                  CREATED          STATUS          PORTS                                           NAMES
54d89d7c4fad   pankb_llm:latest     "streamlit run strea…"   10 minutes ago   Up 10 minutes   0.0.0.0:8501->8501/tcp, :::8501->8501/tcp       pankb-llm
```

## Availability

Currently, the Streamlit app is available as a django application:
```
http://<toedit: pankb server-ip or domain name>/ai_assistant/
```