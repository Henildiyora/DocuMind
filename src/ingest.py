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

class IngestionPipeline:
    '''
    A pipeline to ingest documents into Pinecone vector store.
    '''

    def __init__(self,pinecone_api_key:str = None, index_name:str = "index-v1", embedding_dimension:int = None):

        self.PINECONE_API_KEY = pinecone_api_key
        self.INDEX_NAME = index_name
        self.EMBEDDING_DIMENSION = embedding_dimension
        if not self.PINECONE_API_KEY:
            raise ValueError("API keys not found. Please set them in the .env file.")
        
        # Check the tracking status
        self.langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
        tracing_status = os.getenv("LANGCHAIN_TRACING_V2")
        if tracing_status == "true" and self.langchain_api_key:
            print(" LangSmith Tracing is ENABLED. Your runs will be recorded.")
        else:
            print(" LangSmith Tracing is DISABLED. Add LANGCHAIN_API_KEY to .env to track runs.")
        
        
        # Ensure Index Exists
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        '''
        Ensure that the Pinecone index exists, create it if it does not.
        '''

        # Initilializing pinecone
        pc = Pinecone(api_key=self.PINECONE_API_KEY)

        existing_indexes = [index.name for index in pc.list_indexes()]

        if self.INDEX_NAME not in existing_indexes:
            print(f"Creating index: {self.INDEX_NAME}")
            pc.create_index(
                name=self.INDEX_NAME,
                dimension=self.EMBEDDING_DIMENSION, 
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                )
            )
            print(f"Index {self.INDEX_NAME} created.")
        else:
            print(f"Index {self.INDEX_NAME} already exists.")

    def run(self,file_path:str):
        '''
        Run the ingestion pipeline.
        '''
        # Load the pdfs 
        print("Loading documents")
        raw_docs = self._load_documents(file_path)
        print(f"Loaded {len(raw_docs)} pages.")

        # Split the documents into chunks
        print("Splitting documents into chunks")
        documents = self._split_documents(raw_docs)
        print(f"Split into {len(documents)} chunks.")

        # Store the documents in Pinecone
        print(f"Embedding and Storing on index : {self.INDEX_NAME}")
        self._store_in_pinecone(documents)
        print("Ingestion complete.")

    def _load_documents(self,file_path:str):
        '''
        Load documents from the given file path.
        '''
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    def _split_documents(self,documents):
        '''
        Split documents into smaller chunks.
        '''
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        return text_splitter.split_documents(documents)
    
    def _store_in_pinecone(self,documents):
        '''
        Embed and store documents in Pinecone.
        '''
        # Embed and store the documents in Pinecone
        print("Embedding and storing documents in Pinecone")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=self.INDEX_NAME,
        )


if __name__ == "__main__":

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = "documind-index"
    EMBEDDING_DIMENSION = 384 # We use 384 dimensions because that is the output size of HuggingFace's embedding model.
    FILE_PATH = "data/sample.pdf"  # Path to the PDF file to be ingested.

    pipeline = IngestionPipeline(pinecone_api_key=PINECONE_API_KEY, 
                                 index_name=INDEX_NAME, 
                                 embedding_dimension=EMBEDDING_DIMENSION)
    pipeline.run(file_path=FILE_PATH)
