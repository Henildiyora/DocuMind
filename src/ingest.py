import os 
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_chroma import Chroma

# Load secrets from .env file
load_dotenv()

SUPPORTED_EXTENSIONS = {
    # Code
    ".py", ".js", ".ts", ".jsx", ".tsx", 
    ".java", ".cpp", ".c", ".h", ".cs", 
    ".go", ".rs", ".php", ".rb", ".swift", 
    # Config & Web
    ".json", ".yaml", ".yml", ".toml", ".xml",
    ".html", ".css", ".scss",
    ".sql", ".md", ".txt"
}

IGNORE_DIRS = {
    "venv", ".venv", "env", "DocuMind_venv",
    "__pycache__", ".git", ".idea", ".vscode",
    "node_modules", "site-packages", "dist-packages",
    "build", "dist", "bin", "obj", "target", "include", "lib"
}

class IngestionPipeline:
    '''
    A pipeline to ingest documents into Pinecone vector store.
    '''

    def __init__(self,mode:str, pinecone_api_key:str, index_name:str, embedding_dimension:int):
        self.MODE = mode
        self.PINECONE_API_KEY = pinecone_api_key
        self.INDEX_NAME = index_name
        self.EMBEDDING_DIMENSION = embedding_dimension
        
        if not self.PINECONE_API_KEY:
            raise ValueError("API keys not found. Please set them in the .env file.")
        
        self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        existing_indexes = [index.name for index in self.pc.list_indexes()]

        if self.INDEX_NAME not in existing_indexes:
            print(f"Creating index: {self.INDEX_NAME}")
            self.pc.create_index(
                name=self.INDEX_NAME,
                dimension=self.EMBEDDING_DIMENSION, 
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Index {self.INDEX_NAME} created.")
        else:
            print(f"Index {self.INDEX_NAME} found.")

    def run(self, directory_path:str):
        print(f"\n Starting Ingestion for: {directory_path}")
        
        # Clear old data
        index = self.pc.Index(self.INDEX_NAME)
        try:
            print("Clearing old vector data")
            index.delete(delete_all=True)
        except Exception as e:
            print(f"Warning during delete: {e}")

        # Load documents
        pdf_docs = self._load_documents(directory_path)
        print(f"Loaded {len(pdf_docs)} PDF pages.")

        code_docs = self._load_code(directory_path)
        print(f"Loaded {len(code_docs)} code files.")

        # Split
        chunks = []
        if pdf_docs:
            print("Splitting PDFs")
            chunks.extend(self._split_documents(pdf_docs))
        
        if code_docs:
            print("Splitting Code")
            chunks.extend(self._split_code(code_docs))
            
        print(f"Total chunks created: {len(chunks)}")

        # Store
        if chunks:
            if self.MODE == "ONLINE":
                print(f"Embedding and Storing in Pinecone Index: {self.INDEX_NAME}...")
                self._store_in_pinecone(chunks)
            else:
                print("Embedding and Storing in Local ChromaDB...")
                self._store_local(chunks)
            print("Ingestion complete successfully.")
        else:
            print("No documents found to ingest.")

    def _load_documents(self, directory_path:str):
        docs = []
        if os.path.isfile(directory_path) and directory_path.endswith(".pdf"):
             files = [directory_path]
        else:
             files = glob.glob(f"{directory_path}/**/*.pdf", recursive=True)
             
        for file in files:
            # Skipping ignores
            if any(ign in file for ign in IGNORE_DIRS): 
                continue

            try:
                loader = PyPDFLoader(file_path=file)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source_type"] = "document"
                    doc.metadata["file_path"] = file
                docs.extend(loaded_docs)
            except Exception as e:
                print(f" Error loading document {file}: {e}")
        return docs

    def _load_code(self, directory_path:str):
        docs = []
        if os.path.isfile(directory_path):
             files = [directory_path]
        else:
             files = glob.glob(f"{directory_path}/**/*", recursive=True)

        for file in files:
            # Skip directories
            if os.path.isdir(file): continue

            # Skip ignored folders
            # We split path parts to ensure we don't accidentally match a filename like "env.py"
            parts = file.split(os.sep)
            if any(part in IGNORE_DIRS for part in parts):
                continue

            # Check extension
            _, ext = os.path.splitext(file)
            if ext.lower() not in SUPPORTED_EXTENSIONS:
                continue

            try: 
                loader = TextLoader(file_path=file, encoding="utf-8")
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source_type"] = "code"
                    doc.metadata["file_path"] = file
                    doc.metadata["language"] = ext.replace(".", "")
                docs.extend(loaded_docs)
            except Exception as e:
                print(f" Warning: Could not load {file}: {e}")
        return docs

    def _split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(documents)
    
    def _split_code(self, documents):
        code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=1000, chunk_overlap=200
        )
        return code_splitter.split_documents(documents)
    
    def _store_in_pinecone(self, documents):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=self.INDEX_NAME,
        )
    
    def _store_local(self, documents):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="local_docs",
            persist_directory="./local_chromadb_storage"
        )


if __name__ == "__main__":

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = "documind-code-docs-index"
    EMBEDDING_DIMENSION = 384 # We use 384 dimensions because that is the output size of HuggingFace's embedding model.
    ROOT_DIR = "./"
    MODE = "LOCAL"

    pipeline = IngestionPipeline(mode=MODE, pinecone_api_key=PINECONE_API_KEY, 
                                 index_name=INDEX_NAME, 
                                 embedding_dimension=EMBEDDING_DIMENSION)
    
    pipeline.run(directory_path=ROOT_DIR)