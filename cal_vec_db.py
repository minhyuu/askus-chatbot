import os
import json
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from llama_index.llms.groq import Groq
from llama_index.core import Settings

from dotenv import load_dotenv


load_dotenv()  # load var in .env

api_key = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = api_key

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm_70b = Groq(model="llama3-70b-8192")


Settings.llm = llm_70b
Settings.embed_model = embed_model

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
    embed_model=embed_model
)

index.storage_context.persist()  # save vector store

print("Vector store created and persisted successfully.")
