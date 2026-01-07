import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
import time 

# Load secrets from .env file
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "documind-index-v2"
EMBEDDING_DIMENSION = 384 # We use 384 dimensions because that is the output size of HuggingFace's embedding model.

if not PINECONE_API_KEY:
    raise ValueError("API keys not found. Please set them in the .env file.")


# Initilializing pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [index.name for index in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    print(f"Creating index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION, 
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        )
    )
    print(f"Index {INDEX_NAME} created.")
else:
    print(f"Index {INDEX_NAME} already exists.")


# the pipeline logic would go here
def ingest_docs():

    # Load the pdfs 
    print("Loading documents")
    loader = PyPDFLoader("data/sample.pdf")
    raw_docs = loader.load()
    print(f"Loaded {len(raw_docs)} pages.")

    # Split the documents into chunks
    print("Splitting documents into chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    documents = text_splitter.split_documents(raw_docs)
    print(f"Split into {len(documents)} chunks.")

    # Embed and store the documents in Pinecone
    print("Embedding and storing documents in Pinecone")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    print("Ingestion complete.")
                
if __name__ == "__main__":
    ingest_docs()