#
# The Streamlit app
# Authors: Binhuan Sun (binsun@biosustain.dtu.dk), Pashkova Liubov (liupa@dtu.dk)
#
#

import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain.schema.runnable import RunnableMap
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch, CosmosDBSimilarityType, CosmosDBVectorSearchType
from langchain_voyageai import VoyageAIEmbeddings
import streamlit as st
from pymongo import MongoClient


dotenv.load_dotenv()

# Obtain the MongoDB connection string: ----
connection_string = os.getenv("MONGODB_CONN_STRING")
# The MongoDB database instance name: ----
db_name = "pankb_llm"
# The MongoDB collection name (must be populated with vector embeddings): ----
collection_name = "pankb_vector_store"

# Connect to the MongoDB instance: ----
client = MongoClient(connection_string)
# Obtain the db collection object: ----
collection = client[db_name][collection_name]


def format_docs(docs):
    return "\n\n".join('Title: ' + doc.metadata['title'] + '.' + ' Content: ' + doc.page_content for doc in docs)

def filter_and_extract_documents(documents):
    filtered_documents = [doc for doc in documents if doc.metadata['relevance_score'] >= 0.5]
    return filtered_documents

def get_retriever(db_name, collection_name):
    # Important: here we can use only embedding models with dimensionality up to 2000,
    # because for Azure Cosmos Db for MongoDB the maximum number of supported dimensions is 2000: ----
    embeddings = VoyageAIEmbeddings(model="voyage-large-2-instruct", show_progress_bar=True)

    namespace = db_name + '.' + collection_name
    vectordb = AzureCosmosDBVectorSearch.from_connection_string(connection_string, namespace, embeddings)

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 30})

    compressor = CohereRerank(top_n=20)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever

def question_answer(retriever, question):
    template = """You are PangenomeLLM. You are a cautious assistant proficient in microbial pangenomics. Use the following pieces of context to answer user's questions. 
    Please check the information in the context carefully and do not use information that is not relevant to the question. 
    If the retrieved context doesn't provide useful information to answer user's question, just say that you don't know. 
    Question: {question}
    Context: {context}
    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0, model_kwargs={"top_p": 0.0})

    rag_chain_from_docs = (
            {
                "context": lambda input: format_docs(input["documents"]),
                "question": itemgetter("question"),
            }
            | prompt
            | llm
    )

    rag_chain_with_source = RunnableMap(
        {"documents": retriever, "question": RunnablePassthrough()}
    ) | RunnableMap(
        {"documents": lambda input: filter_and_extract_documents(input['documents']),
         "question": lambda input: input["question"]}
    ) | {
                                "answer": rag_chain_from_docs,
                                "Source": lambda input: list(set(doc.metadata['source'] for doc in input["documents"])),
                            }
    response = rag_chain_with_source.invoke(question)

    answer = response['answer'].content + '<br>' + '<br>' + '<b>Reference:</b>' + '<br>'

    # Generate links with numbers, assuming the list items are URLs
    links = [f'<a href="{url}">paper {i + 1}</a>' for i, url in enumerate(response['Source'])]

    # Join the links into a single string separated by commas
    formatted_links = '<br>'.join(links)

    # Append the formatted links to the answer
    answer += formatted_links

    return answer


if __name__ == "__main__":
    st.title("PanKB LLM: Your Microbial Pangenomics Assistant")

    # Initialize your retriever just once
    retriever = get_retriever(db_name, collection_name)

    # Initialize session state for holding messages if it does not exist
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display old messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    # Handle new input from the user
    if prompt := st.chat_input("What is your question about microbial pangenomics?"):
        # Add user's question to the session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner('Generating answer...'):
            answer = question_answer(retriever, prompt)

        # Add the assistant's response to the session state
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Display the assistant's response immediately
        with st.chat_message("assistant"):
            st.markdown(answer, unsafe_allow_html=True)