import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = CSVLoader("./Data/yahoo_apple_news.csv")

data = loader.load()


text_split_model = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 50,
    separators=["\n"]
)

splited_document = text_split_model.split_documents(data)


print(len(splited_document))

