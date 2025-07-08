from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import os
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from dotenv import load_dotenv
from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.core.retrievers import VectorIndexRetriever, BM25Retriever, HybridRetriever
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

load_dotenv()  # load var in .env

api_key = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = api_key

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm_70b = Groq(model="llama3-70b-8192")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

Settings.llm = llm_70b
Settings.embed_model = embed_model

# Reload Chroma collection
CHROMA_DIR = "chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
chroma_collection = chroma_client.get_or_create_collection("my_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Create StorageContext
storage_context = StorageContext.from_defaults(
    persist_dir="storage",
    vector_store=vector_store
)

# Reload index from saved storage context
index = load_index_from_storage(storage_context)

# Create query engine
# query_engine = index.as_query_engine(top_k=3)
vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=4)

query_transform = HyDEQueryTransform(llm=llm_70b)

synthesizer = TreeSummarize(llm=llm_70b)

query_engine = RetrieverQueryEngine.from_args(
    retriever=vector_retriever,
    response_synthesizer=synthesizer,
    query_transform=query_transform,
    citation_chunk_size=512,
    citation_retriever=vector_retriever,  # dùng để truy xuất citation
)

while True:
    current_question = input("Enter your query (or 'exit' to quit): ")
    if current_question.lower() == 'exit':
        break
    

    response = query_engine.query(current_question)
    print("🤖 Bot:")
    print(response)
    
    # print("source:", response.source_nodes[0].metadata['url'])
    print("\nSource:")
    for node in response.source_nodes:
        url = node.metadata.get("url", "No URL")
        print(f"- {url}")





