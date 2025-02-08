import streamlit as st
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import embeddings
from .file_reader import process_file


class VectorStore():
    def __init__(self):
        self.vector_store = InMemoryVectorStore(embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


    def create_vector_store(self):
        content = ""

        file = st.file_uploader(
            label="Upload a PDF file",
            accept_multiple_files=False,
            type=['pdf']
            )
        
        if file is not None:
            content = process_file(file)
            
        all_splits = self.text_splitter.split_text(content)

        _ = self.vector_store.add_texts(texts=all_splits)

    def get_docs(self, query, num_of_docs=4):
        docs = self.vector_store.similarity_search(query=query, k=num_of_docs)
        return docs