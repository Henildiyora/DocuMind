"""
Ingestion Pipeline Module

Handles loading, splitting, and storing documents/code into vector databases
for DocuMind's knowledge base.
"""

import os
import glob
from typing import List, Optional
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import (
    PINECONE_API_KEY,
    INDEX_NAME,
    EMBEDDING_DIMENSION,
    LOCAL_VECTOR_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SUPPORTED_EXTENSIONS,
    IGNORE_DIRS,
    EMBEDDING_MODEL,
    logger
)


class IngestionPipeline:
    """
    A pipeline to ingest documents into Pinecone or ChromaDB vector store.

    Supports both online (Pinecone) and local (ChromaDB) modes for storing
    code and document embeddings.
    """

    def __init__(
        self,
        mode: str = "LOCAL",
        pinecone_api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        embedding_dimension: Optional[int] = None
    ) -> None:
        """
        Initialize the ingestion pipeline.

        Args:
            mode: Storage mode ("LOCAL" or "ONLINE").
            pinecone_api_key: API key for Pinecone (required for ONLINE mode).
            index_name: Name of the vector index.
            embedding_dimension: Dimension of embeddings.

        Raises:
            ValueError: If required parameters are missing.
        """
        self.mode = mode.upper()
        self.pinecone_api_key = pinecone_api_key or PINECONE_API_KEY
        self.index_name = index_name or INDEX_NAME
        self.embedding_dimension = embedding_dimension or EMBEDDING_DIMENSION

        if self.mode == "ONLINE" and not self.pinecone_api_key:
            raise ValueError("Pinecone API key required for ONLINE mode")

        self.pc: Optional[Pinecone] = None
        if self.mode == "ONLINE":
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            self._ensure_index_exists()

        logger.info(f"Initialized IngestionPipeline in {self.mode} mode")

    def _ensure_index_exists(self) -> None:
        """Ensure the Pinecone index exists, creating it if necessary."""
        if not self.pc:
            return

        existing_indexes = [index.name for index in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        else:
            logger.info(f"Pinecone index {self.index_name} already exists")

    def run(self, directory_path: str) -> None:
        """
        Run the full ingestion pipeline.

        Args:
            directory_path: Path to directory or file to ingest.
        """
        logger.info(f"Starting ingestion for: {directory_path}")

        # Clear old data
        if self.mode == "ONLINE" and self.pc:
            index = self.pc.Index(self.index_name)
            try:
                logger.info("Clearing old vector data from Pinecone")
                index.delete(delete_all=True)
            except Exception as e:
                logger.warning(f"Warning during Pinecone delete: {e}")
        elif self.mode == "LOCAL":
            # ChromaDB handles persistence automatically
            pass

        # Load documents
        pdf_docs = self._load_documents(directory_path)
        logger.info(f"Loaded {len(pdf_docs)} PDF pages")

        code_docs = self._load_code(directory_path)
        logger.info(f"Loaded {len(code_docs)} code files")

        # Split documents
        chunks: List[Document] = []
        if pdf_docs:
            logger.info("Splitting PDF documents")
            chunks.extend(self._split_documents(pdf_docs))

        if code_docs:
            logger.info("Splitting code documents")
            chunks.extend(self._split_code(code_docs))

        logger.info(f"Total chunks created: {len(chunks)}")

        # Store chunks
        if chunks:
            if self.mode == "ONLINE":
                logger.info(f"Storing in Pinecone index: {self.index_name}")
                self._store_in_pinecone(chunks)
            else:
                logger.info("Storing in local ChromaDB")
                self._store_local(chunks)
            logger.info("Ingestion completed successfully")
        else:
            logger.warning("No documents found to ingest")

    def _load_documents(self, directory_path: str) -> List[Document]:
        """
        Load PDF documents from the given path.

        Args:
            directory_path: Path to scan for PDFs.

        Returns:
            List of loaded documents with metadata.
        """
        docs: List[Document] = []
        if os.path.isfile(directory_path) and directory_path.endswith(".pdf"):
            files = [directory_path]
        else:
            files = glob.glob(f"{directory_path}/**/*.pdf", recursive=True)

        for file_path in files:
            if any(ign in file_path for ign in IGNORE_DIRS):
                continue

            try:
                loader = PyPDFLoader(file_path=file_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source_type"] = "document"
                    doc.metadata["file_path"] = file_path
                docs.extend(loaded_docs)
            except Exception as e:
                logger.warning(f"Error loading document {file_path}: {e}")

        return docs

    def _load_code(self, directory_path: str) -> List[Document]:
        """
        Load code files from the given path.

        Args:
            directory_path: Path to scan for code files.

        Returns:
            List of loaded code documents with metadata.
        """
        docs: List[Document] = []
        if os.path.isfile(directory_path):
            files = [directory_path]
        else:
            files = glob.glob(f"{directory_path}/**/*", recursive=True)

        for file_path in files:
            if os.path.isdir(file_path):
                continue

            parts = file_path.split(os.sep)
            if any(part in IGNORE_DIRS for part in parts):
                continue

            _, ext = os.path.splitext(file_path)
            if ext.lower() not in SUPPORTED_EXTENSIONS:
                continue

            try:
                loader = TextLoader(file_path=file_path, encoding="utf-8")
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source_type"] = "code"
                    doc.metadata["file_path"] = file_path
                    doc.metadata["language"] = ext.replace(".", "")
                docs.extend(loaded_docs)
            except Exception as e:
                logger.warning(f"Warning: Could not load {file_path}: {e}")

        return docs

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split document chunks.

        Args:
            documents: Documents to split.

        Returns:
            List of document chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        return text_splitter.split_documents(documents)

    def _split_code(self, documents: List[Document]) -> List[Document]:
        """
        Split code documents with language-aware splitting.

        Args:
            documents: Code documents to split.

        Returns:
            List of code chunks.
        """
        code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        return code_splitter.split_documents(documents)

    def _store_in_pinecone(self, documents: List[Document]) -> None:
        """
        Store documents in Pinecone vector store.

        Args:
            documents: Documents to store.
        """
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=self.index_name,
        )

    def _store_local(self, documents: List[Document]) -> None:
        """
        Store documents in local ChromaDB.

        Args:
            documents: Documents to store.
        """
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=LOCAL_VECTOR_DIR
        )