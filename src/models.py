import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings


def get_llm(GOOGLE_API_KEY: str = None) -> ChatGoogleGenerativeAI:
    llm = ChatGoogleGenerativeAI(
        model=os.getenv('GEMINI_MODEL'),
        api_key=GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY"),
        temperature=os.getenv('TEMPERATURE')
    )
    return llm

embeddings = GoogleGenerativeAIEmbeddings(
    model=os.getenv('EMBEDDINGS_MODEL')
)