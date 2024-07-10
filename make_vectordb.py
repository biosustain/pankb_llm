#
# The script builds a vector database out of txt documents
# based on an Azure Cosmos DB for MongoDB instance within a sharded cluster.
# Authors: Binhuan Sun (binsun@biosustain.dtu.dk), Pashkova Liubov (liupa@dtu.dk)
#

import os
import dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch, CosmosDBSimilarityType, CosmosDBVectorSearchType
from langchain_voyageai import VoyageAIEmbeddings
import pandas as pd
import argparse
import time
from pymongo import MongoClient


dotenv.load_dotenv()


def process_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Extract URL and Title
    doi = lines[0].strip()
    title = lines[1].replace('title:', '').replace('Title:', '').strip()

    # Extract page content after "TRANSCRIPT"
    page_content = ''.join(lines[2:])

    return Document(page_content=page_content, metadata={'source': doi, 'title': title})


def create_documents_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            doc = process_txt_file(os.path.join(directory_path, filename))
            documents.append(doc)
    return documents


def split_list(input_list, chunk_size):
    """Yield successive n-sized chunks from input_list."""
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i + chunk_size]



if __name__ == "__main__":
    script_start_time = time.time()
    # Get plain text dir, collection name and vector database path from command line
    parser = argparse.ArgumentParser(description='Get collection name and vector database path')
    parser.add_argument('plain_text_dir', type=str, help='The path to your plain text files')
    parser.add_argument('collection', type=str, help='The name of the collection of your vector database')
    args = parser.parse_args()

    plain_text_dir = str(args.plain_text_dir)
    collection_name = str(args.collection)

    directory_path = plain_text_dir

    # Create documents from the directory
    docs = create_documents_from_directory(directory_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    print([directory_path, len(docs), len(all_splits)])

    # QC-remove duplicates
    df_qc = pd.DataFrame({'all_splits': all_splits, 'content': [doc.page_content for doc in all_splits]})
    df_qc_no_dup = df_qc.drop_duplicates(subset='content', keep='first')
    all_splits_no_dup = df_qc_no_dup['all_splits'].tolist()
    print('The final length of all_split: ' + str(len(all_splits_no_dup)))

    # Start to build the vectordb...
    # Important: here we can use only embedding models with dimensionality up to 2000,
    # because for Azure Cosmos Db for MongoDB the maximum number of supported dimensions is 2000: ----
    embeddings = VoyageAIEmbeddings(model="voyage-large-2-instruct", show_progress_bar=True)

    split_docs_chunked = split_list(all_splits_no_dup, 41000)

    # Obtain the MongoDB connection string: ----
    connection_string = os.getenv("MONGODB_CONN_STRING")
    # The MongoDB database instance name: ----
    db_name = "pankb_llm"
    # Set the name of the db index to be created: ----
    index_name = "pankb_vector_store_hnsw_index"

    # Connect to the MongoDB instance: ----
    client = MongoClient(connection_string)
    # Obtain the db collection object: ----
    collection = client[db_name][collection_name]

    # Drop the MongoDB collection if it exists: ----
    collection.drop()

    print("Creating the vector index...")
    # Note: The index is created before we insert the data
    # (despite it increases the total data insertion time, as the index has to be updated after each insertion).
    # We create it beforehand to avoid timeout errors,
    # which we get if we try to create the index after populating the Vector DB and which
    # we can not get rid of (presumably due to the cluster hardware configuration can not be changed).
    #
    # Example of the same problem enciuntered while creating the index on the M40 tier is described here:
    # https://stochasticcoder.com/2024/03/08/azure-cosmos-db-for-mongodb-hnsw-vector-search/
    #
    client[db_name].command({
        "createIndexes": collection_name,
        "indexes": [
            {
                "name": collection_name + "_hnsw_index",
                "key": {
                    "vectorContent": "cosmosSearch"
                },
                "cosmosSearchOptions": {
                    "kind": "vector-hnsw",
                    "m": 16,
                    "efConstruction": 100,
                    "similarity": "L2",
                    "dimensions": 1024
                }
            }
        ]
    })
    print("The vector index has been successfully created.")

    # Now populate the database updating the index in the meantime: ----
    print("Populating the Vector DB (can take a while)...")
    for split_docs_chunk in split_docs_chunked:
        vectordb = AzureCosmosDBVectorSearch.from_documents(
            split_docs_chunk,
            embeddings,
            collection=collection,
            index_name=index_name)

    print("Total execution time: %.2f minutes" % ((time.time() - script_start_time) / 60))