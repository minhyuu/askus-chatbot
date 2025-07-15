import streamlit as st
import os
from dotenv import load_dotenv

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


# load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = api_key


os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
def create_chat_engine(_index):

    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

    return _index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=(
        "You are a helpful and professional virtual assistant for the University of Tasmania (UTAS). "
        "You specialize in answering questions about topics such as enrollment, fees, timetables, scholarships, login and password help at UTAS. "
        "Your answers must be accurate and based strictly on the provided knowledge base. "
        "If you are unsure or don't know the answer, clearly say so. For example: 'I'm not sure about that, but you can contact UConnect for further assistance.' "
        "Avoid making up information. Do not guess unless you make it clear that it is only a general suggestion. For example: 'This is just my suggestion based on common practice at UTAS...' "
        "If users ask about general topics or make casual small talk (e.g., 'Who are you?', 'What can you do?', 'How are you?'), respond in a natural, friendly, and conversational tone like a chatbot companion. "
        "If users ask questions outside your area of expertise, politely let them know. You can still engage in a helpful way, and recommend relevant UTAS support services such as:\n"
        "- UConnect (for general support and inquiries)\n"
        "- Student Advisers (for study-related questions and academic support)\n"
        "- Learning Advisers (for help with academic writing, assignments, and learning strategies)"
        )
    )


def main():

    # UI
    st.title("🧠 AskUs Chatbot (RAG Model)")

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
    👋 Hi! I'm your friendly assistant for common questions at the **University of Tasmania (UTAS)**.

    I specialise in answering frequently asked questions about:
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

    RELEVANCE_THRESHOLD = 0.5  # threshold to filter out low relevance sources

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_engine.chat(prompt)

                relevant_sources = [
                    node.metadata.get("url")
                    for node in response.source_nodes or []
                    if node.metadata.get("url") and node.score and node.score > RELEVANCE_THRESHOLD
                ]

                if relevant_sources:
                    response_text = f"{response.response}\n\n**Sources:**\n" + "\n".join(relevant_sources)
                else:
                    # if no relevant sources, provide a revised response
                    revised_prompt = (
                        f"You are an assistant restricted to only provide answers based on the knowledge base. "
                        f"The retrieved sources has low relevant level"
                        f"Please revise your answer if user query related to your expertise. "
                        f"If users ask about general topics or make casual small talk (e.g., 'Who are you?', 'What can you do?', 'How are you?'), respond in a natural, friendly, and conversational tone like a chatbot companion. "
                    )
                    revised_response = chat_engine.chat(revised_prompt)
                    response_text = revised_response.response

                st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})



    # Inject custom CSS for footer
    st.markdown("""
        <style>
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: rgba(248, 249, 250, 0); /* semi-transparent */
                color: #6c757d;
                text-align: center;
                padding: 5px 0;
                font-size: 0.8rem;
                z-index: 9999;
            }
        </style>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <div class="footer">
            Chat bot may not always provide accurate information.
        </div>
        """, 
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
