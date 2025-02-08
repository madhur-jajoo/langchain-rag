import streamlit as st

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.base import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

from .vector_store import VectorStore

class ChatAgent():
    def __init__(self, llm: Runnable, prompt: ChatPromptTemplate):
        """This class is used to initialize the chat agent

        Args:
            - llm (Runnable): LLM to be used.
            - prompt (ChatPromptTemplate): The RAG prompt to be used.
        """
        self.history = StreamlitChatMessageHistory(key="chat_messages")
        self.llm = llm
        self.prompt = prompt
        self.chain = self.setup_chain()

    def setup_chain(self) -> RunnableWithMessageHistory:
        """Setup the chain for the ChatAgent

        Returns:
            RunnableWithMessageHistory: The configured chain with message history
        """
        chain = self.prompt | self.llm

        return RunnableWithMessageHistory(
            chain,
            lambda session_id: self.history,
            input_messages_key="question",
            history_messages_key="history"
        )
    
    def display_messages(self):
        """
        Display the session message in the chat interface.

        If no message is present, then add a default AI message.
        """
        if len(self.history.messages) == 0:
            self.history.add_ai_message("I am a document expert. How can I help you?")

        for message in self.history.messages:
            st.chat_message(message.type).write(message.content)

    def start_conversation(self, vector_store: VectorStore):
        self.display_messages()
        
        if user_question := st.chat_input(placeholder="Type a message..."):
            st.chat_message("human").write(user_question)
            config = {"configurable": {"session_id": "any"}}
            context = vector_store.get_docs(query=user_question)
            response = self.chain.invoke(input={"question": user_question, "context": context}, config=config)
            st.chat_message("ai").write(response.content)

    def reset_history(self) -> None:
        """
        Reset the chat history to start a new session
        """
        self.history.clear()