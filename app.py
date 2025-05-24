import os
import streamlit as st
import requests
from dotenv import load_dotenv
from utils.chat_engine import ChatEngine

# Load environment variables
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Setup Streamlit page
st.set_page_config(page_title="Bank Customer Service Chatbot", layout="wide")
st.title("🏦 Bank Assistant Chatbot")

# Define local and remote vector paths
local_vector_path = "vectors/index.pkl"
github_raw_url = "https://raw.githubusercontent.com/Addisu-Taye/bank-customer-service-chatbot/main/vectors/index.pkl"

# Ensure vector file exists locally
if not os.path.exists(local_vector_path):
    st.info("Downloading vector database from GitHub...")
    os.makedirs(os.path.dirname(local_vector_path), exist_ok=True)
    response = requests.get(github_raw_url)
    if response.status_code == 200:
        with open(local_vector_path, "wb") as f:
            f.write(response.content)
        st.success("Vector database downloaded.")
    else:
        st.error("Failed to download vector data from GitHub.")
        st.stop()

# Initialize ChatEngine once
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = ChatEngine(local_vector_path)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display past chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about our bank services..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    response = st.session_state.chat_engine.chat(prompt)

    st.chat_message("assistant").markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
