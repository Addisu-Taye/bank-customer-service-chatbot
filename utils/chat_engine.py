import pickle
from textblob import TextBlob
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings


class ChatEngine:
    def __init__(self, vector_path: str):
        """Initialize Chat Engine with vector DB and QA chain."""
        with open(vector_path, "rb") as f:
            self.vectordb = pickle.load(f)

        retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})

        # Setup HuggingFace transformers pipeline locally
        pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            max_length=512,
            temperature=0.3,
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
        )

    def correct_spelling(self, query: str) -> str:
        """Spell-check and return corrected query."""
        return str(TextBlob(query).correct())

    def chat(self, query: str) -> str:
        """Answer user queries using QA chain."""
        corrected_query = self.correct_spelling(query)
        print(f"[Corrected] {query} → {corrected_query}")  # Optional logging
        return self.qa_chain.invoke({"query": corrected_query})


def get_vector_store(persist_directory="vectors"):
    """Load FAISS vector store using embeddings."""
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(persist_directory, embedding)
