import os
import streamlit as st
from dotenv import load_dotenv
from utils.chat_engine import ChatEngine

# Load environment variables
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Setup Streamlit page
st.set_page_config(page_title="Bank Customer Service Chatbot", layout="wide")
st.title("🏦 Bank Assistant Chatbot")

# Initialize ChatEngine once
if "chat_engine" not in st.session_state:
    vector_path = "vectors/bank_data.pkl"
    st.session_state.chat_engine = ChatEngine(vector_path)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display past chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about our bank services..."):
    # Add user message
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Get response from ChatEngine
    response = st.session_state.chat_engine.chat(prompt)

    # Add assistant response
    st.chat_message("assistant").markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
