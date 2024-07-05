# PanKB LLM (PRE-PROD)

## Overview

A GenAI Assistant based on Langchain + Streamlit + Azure Cosmos DB for MongoDB (vCore) + Docker.

Authors:
- Data preprocessing, LLM, DEV vector DB creation (Chroma), retriever, Streamlit web app: Binhuan Sun, binsun@biosustain.dtu.dk 
- Changing the DEV vector DB (Chroma) to the PROD vector DB instance (Azure Cosmos DB for MongoDB (vCore)) and adjusting the choice of embeddings to the Cosmos DB limitations, the PROD DB index creation, dockerization, integration of the streamlit app with the Django framework templates, the github repo maintenance: Pashkova Liubov, liupa@dtu.dk

## Scripts execution

Every time when one pushes to the `pre-prod` repo (usually from the DEV server), the changes in the AI Assistant Web Application will be AUTOMATICALLY deployed to the PRE-PROD server. The automation (CI/CD) is achieved with the help of Github Actions enabled for the repository. The respective config file is `.github/workflows/deploy-preprod-to-azurevm.yml`. In order for the automated deployment to work, you should set up the values of the following secret Github Actions secrets:
```
PANKB_PREPROD_HOST - the PRE-PROD server IP address
PANKB_PREPROD_SSH_USERNAME - the ssh user name to connect to the PRE-PROD server
PANKB_PREPROD_PRIVATE_SSH_KEY - the ssh key that is used to connect to the PRE-PROD server
OPENAI_API_KEY
COHERE_API_KEY
TOGETHER_API_KEY
GOOGLE_API_KEY
ANTHROPIC_API_KEY
REPLICATE_API_TOKEN
VOYAGE_API_KEY
PANKB_PREPROD_MONGODB_CONN_STRING - MongoDB PRE-PROD (Azure CosmosDB for MongoDB) Connection String
```
These secrets are encrypted and safely stored on Github in the "Settings - Secrets and Variables - Actions - Repository secrets" section. In this section, you can also add new Github Actions secrets and edit the existing ones. However, in order to change a secret name, you have to remove the existing secret and add the new one instead of the old one.

After the Github Actions deployment job has successfully run, the web-application must be available at <a href="pankb.org/ai_assistant" target="_blank">pankb.org/ai_assistant</a>. 

The status of the docker container can be checked with the following command:
```
docker ps
```
The command should produce approx. the following output in case of the successful deployment:
```
CONTAINER ID   IMAGE                COMMAND                  CREATED          STATUS          PORTS                                           NAMES
54d89d7c4fad   pankb_llm:latest     "streamlit run strea…"   23 seconds ago   Up 12 seconds   0.0.0.0:8501->8501/tcp, :::8501->8501/tcp       pankb-llm
```