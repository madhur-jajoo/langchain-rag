import streamlit as st

from src.chat_utils import ChatAgent
from src.vector_store import VectorStore
from src.models import get_llm
from src.prompts import rag_prompt


def main():

    st.set_page_config(
        page_title="📝 RAG Q&A Application",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.madhurjajoo.com/',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# This is a simple app to demonstrate how to build a Streamlit app"
        }
    )
    
    with st.sidebar:
        GOOGLE_API_KEY = st.text_input(label="GOOGLE_API_KEY", type="password")

    vector_store = VectorStore()
    vector_store.create_vector_store()

    chat_agent = ChatAgent(llm=get_llm(GOOGLE_API_KEY), prompt=rag_prompt)
    chat_agent.start_conversation(vector_store=vector_store)

if __name__ == '__main__':
    
    from dotenv import load_dotenv
    load_dotenv(dotenv_path='.env', verbose=True)

    main()