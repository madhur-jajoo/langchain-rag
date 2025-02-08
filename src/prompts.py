import os

from langchain import hub

rag_prompt = hub.pull(os.getenv("RAG_PROMPT"))