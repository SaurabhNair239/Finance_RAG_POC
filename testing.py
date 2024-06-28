import os
import streamlit as st
import logging
from langchain.document_loaders import CSVLoader
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
import warnings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker


warnings.filterwarnings("ignore")

load_dotenv()  # take environment variables from .env (especially openai api key)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")


logging.basicConfig(level=logging.INFO)

st.title("Finance RAG system")

### When its an need to work with news articles via links 

#st = st.empty()

##Embedding model initialisation
#embedding_model = SentenceTransformerEmbeddings(model_name="jinaai/jina-embeddings-v2-small-en")

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vstore = AstraDBVectorStore(
        collection_name="finance",
        embedding=embedding_model,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_API_ENDPOINT,)
if st.button("Load and Preprocess data"):
    #initialising the Database connection
    


    #Loading the data 
    data_loader = CSVLoader("./Data/Yahoo_apple_news.csv")
    st.text("preprocessing Data....✅✅✅")
    data = data_loader.load()
    logging.info("Data loaded successfully")


    #reason for not using links are because of crawler disabled by mostly all the websited and using last 20 records because of low processing power

    document = data[-20:]

    ##splitting data into chunks
    #text_splitter = RecursiveCharacterTextSplitter(
    #        chunk_size=1000,
    #        chunk_overlap = 200
    #    )s

    text_splitter = SemanticChunker(
        OpenAIEmbeddings(model="text-embedding-3-small"), breakpoint_threshold_type="percentile"
    )

    splitted_document = text_splitter.create_documents([d.page_content for d in document])

    st.text("Text Splitter...Started...✅✅✅")
    splitted_document = text_splitter.split_documents(splitted_document)
    logging.info("Data split finished")


    # Saving embeddings to AstraDB 
        
    #Formatting data so that it can be inserted properly in AstraDB
    docs = []
    for entry in splitted_document:
        metadata = {"source": entry.metadata}
        doc = Document(page_content=entry.page_content, metadata=metadata)
        docs.append(doc)

    vstore.add_documents(docs)

    logging.info("Embedding finished and data injected to AstraDB")
    st.text("Embedding Vector Started Building...✅✅✅")

#asking query to the chatbot
query = st.chat_input("Question: ")
if query:

    #retriving top 4 docs as per the query
    retriever = vstore.as_retriever(search_kwargs={"k": 1})

    #prompt template for the ChatPrompt
    prompt_template = """
    Answer the question based only on the supplied context and show some mathematical calcualtions if needed. If you don't know the answer, say you don't know the answer.
    Context: {context}
    Question: {question}
    Your answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatOpenAI(
        model = "gpt-3.5-turbo-0125",
        temperature = 0.9,
    )

    logging.info("OpenAI initiated")
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    

    ##retriving data from astradb and 
    result = chain.invoke(query)
    
    st.header("Answer")
    
    st.write(result)
