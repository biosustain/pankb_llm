#
# The script builds a vector database out of txt documents
# based on an Azure Cosmos DB for MongoDb instance within a sharded cluster.
# Authors: Binhuan Sun (binsun@biosustain.dtu.dk), Pashkova Liubov (liupa@dtu.dk)
#
#
# Usage:
# python make_vectordb_cosmosdb.py ./Paper_all <collection_name>
# In our particular case: ----
# python make_vectordb_cosmosdb.py ./Paper_all pankb_vector_store
# where pankb_vector_store is the name of the MongoDB collection.
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
    index_name = "pankb_vector_store_ivf_index"

    # Connect to the MongoDB instance: ----
    client = MongoClient(connection_string)
    # Obtain the db collection object: ----
    collection = client[db_name][collection_name]

    # Drop the MongoDB collection if it exists: ----
    collection.drop()

    # Populate the MongoDB chunk by chunk: ----
    for split_docs_chunk in split_docs_chunked:
        vectordb = AzureCosmosDBVectorSearch.from_documents(
            split_docs_chunk,
            embeddings,
            collection=collection,
            index_name=index_name)   #here we add the creation of the similarity index

    # Below we set the variables used to construct the DB index: ----
    # Read more about these variables in detail here:
    # https://learn.microsoft.com/en-us/azure/cosmos-db/mongodb/vcore/vector-search
    # num_lists - "This integer is the number of clusters that the inverted file (IVF) index uses to group the vector data.
    # We recommend that numLists is set to documentCount/1000 for up to 1 million documents and to sqrt(documentCount) for more than 1 million documents.
    # Using a numLists value of 1 is akin to performing brute-force search, which has limited performance."
    num_lists = round(len(all_splits_no_dup)/1000)
    # The dimension of the embeddings (must coincide with the dimension of the model chosen, in our case it is "voyage-large-2-instruct" with the dim = 1024)
    dim = 1024
    # Use cosine similarity as the similarity algorithm: ----
    similarity_algorithm = CosmosDBSimilarityType.COS
    # kind - the type of index to be created.
    # The CPU (M30) on a server, where we have our Azure Cosmos DB for MongoDB instance, supports only the vector-ivf index type.
    # To create the vector-hnsw index, we need to upgrade to M40 tier
    # (it will cost us 780.42 USD per month instead of 211.36 that we pay for M30 now).
    kind = CosmosDBVectorSearchType.VECTOR_IVF

    print("Creating the vector index...")
    vectordb.create_index(num_lists, dim, similarity_algorithm, kind)
    print("The index has been successfully created.")
    print("Total execution time: %.2f minutes" % ((time.time() - script_start_time) / 60))