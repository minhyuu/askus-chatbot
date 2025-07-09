import streamlit as st
import os
from dotenv import load_dotenv

# Force pysqlite3 to override sqlite3
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import chromadb

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.response_synthesizers import TreeSummarize
# from llama_index.core.chat_engine import CondenseQuestionChatEngine

from llama_index.core.memory import ChatMemoryBuffer

from llama_index.vector_stores.chroma import ChromaVectorStore

import json

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Document



# load api key from secrets
# api_key = st.secrets["GROQ_API_KEY"]
# os.environ["GROQ_API_KEY"] = api_key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = api_key


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# CHROMA_PATH = "chroma_db"
# INDEX_PATH = "storage"
# COLLECTION_NAME = "my_collection"

@st.cache_resource
def load_llm():
    return Groq(model="llama3-70b-8192")

@st.cache_resource
def load_embed_model():
    return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

@st.cache_resource
def create_index(_embed_model):
    """
    Create and return a VectorStoreIndex.
    This function is cached to avoid re-creating the index on every run.
    """
    
    # Chroma vector store
    CHROMA_DIR = "chroma_db"
    if not os.path.exists(CHROMA_DIR):
        os.makedirs(CHROMA_DIR)

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection("my_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # storage context
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store)

    # Load knowledge base
    with open("knowledge_base.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # convert data to Document and embed
    documents = []
    for entry in raw_data:
        doc = Document(
            text=entry['content'],
            metadata={
                "title": entry['title'], 
                "url": entry['source']
            }
        )
        documents.append(doc)

    # create index and persist
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=_embed_model
    )
    
    return index

@st.cache_resource
def create_query_engine(_llm, _embed_model, _index):
    vector_retriever = VectorIndexRetriever(index=_index, similarity_top_k=2)
    query_transform = HyDEQueryTransform(llm=_llm)
    synthesizer = TreeSummarize(llm=_llm)

    return RetrieverQueryEngine.from_args(
        retriever=vector_retriever,
        response_synthesizer=synthesizer,
        query_transform=query_transform,
        citation_chunk_size=1024,
        citation_retriever=vector_retriever,
    )



def create_chat_engine(index):

    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

    return index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=(
              "You are a helpful and friendly virtual assistant for the University of Tasmania (UTAS). "
        "You specialize in answering questions about enrollment, fees, timetables, login and password at UTAS. "
        "For these topics, you should answer clearly and professionally based on your knowledge base. "
        "If the user asks general or casual questions (e.g., 'Who are you?', 'What can you do?', 'How are you?'), "
        "respond in a natural, friendly tone like a chatbot companion. "
        "If the question is unrelated to your area of expertise and you don’t know the answer, it’s okay to say 'I'm not sure about that.'"
    ),
    )


def main():

    # UI
    st.title("🧠 UniMate Chatbot (RAG Model)")

    llm = load_llm()
    embed_model = load_embed_model()

    Settings.llm = llm
    Settings.embed_model = embed_model

    # Ensure chat_history is initialized before any access
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    index = create_index(embed_model)
    # query_engine = create_query_engine(llm, embed_model, index)
    chat_engine = create_chat_engine(index)


    intro_message = f"""
    👋 Hi! I'm **UniMate** — your friendly assistant for common questions at the **University of Tasmania (UTAS)** 

    I specialize in answering frequently asked questions about:
    - 📚 **Enrolment**
    - 💰 **Fees and payment options**
    - 🗓️ **Timetables**
    - 🔐 **Login and password help**

    My knowledge comes from the official Ask Us - UTAS portal:  
    👉 [Ask Us - UTAS](https://askus.utas.edu.au/)

    Not sure what to ask? Here are some examples:
    - *“How do I enrol in my units?”*
    - *“Where can I find my class timetable?”*
    - *“What should I do if I forgot my password?”*
    - *“When is the fee payment deadline?”*

    If your question is related to these topics, I’ll give you an accurate answer — with source links when available.  
    Otherwise, I’ll do my best to help or let you know if I don’t have the info yet.

    Ready when you are — what would you like to know?

        """
    
    # Greet the user only once
    if "greeted" not in st.session_state:
        st.markdown(intro_message)  # where intro_message is the string above
        st.session_state.greeted = True

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask me something...")

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                # response = chat_engine.chat(prompt)

                # RELEVANCE_THRESHOLD = 0.5  # tweak this value as needed

                # relevant_sources = [
                #     node.metadata.get("url")
                #     for node in response.source_nodes
                #     if node.metadata.get("url") and node.score and node.score > RELEVANCE_THRESHOLD
                # ]

                # if relevant_sources:
                #     response_text = f"{response.response}\n\nSources:\n" + "\n".join(relevant_sources)
                # else:
                #     response_text = response.response

                response_text = chat_engine.chat(prompt)
            
                # response_text = str(custom_response)
                st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
