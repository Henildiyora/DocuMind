# config.py - Centralized configuration for DocuMind
import os
import logging
from typing import Dict, Set

# API Keys and Environment
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Vector Store and Indexing
INDEX_NAME: str = os.getenv("INDEX_NAME", "documind-code-docs-index")
EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
LOCAL_VECTOR_DIR: str = os.getenv("LOCAL_VECTOR_DIR", "./local_chromadb_storage")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "local_docs")

# Model Configurations
DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gemini-2.5-flash")
LOCAL_LLM_MODEL: str = os.getenv("LOCAL_LLM_MODEL", "llama3.2")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Ingestion Settings
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_RETRIEVAL_RESULTS: int = int(os.getenv("MAX_RETRIEVAL_RESULTS", "5"))

# Chat Settings
MAX_CHAT_HISTORY: int = int(os.getenv("MAX_CHAT_HISTORY", "10"))

# Supported File Extensions
SUPPORTED_EXTENSIONS: Set[str] = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".cpp", ".c", ".h", ".cs",
    ".go", ".rs", ".php", ".rb", ".swift",
    ".json", ".yaml", ".yml", ".toml", ".xml",
    ".html", ".css", ".scss",
    ".sql", ".md", ".txt"
}

# Directories to Ignore
IGNORE_DIRS: Set[str] = {
    "venv", ".venv", "env", "DocuMind_venv",
    "__pycache__", ".git", ".idea", ".vscode",
    "node_modules", "site-packages", "dist-packages",
    "build", "dist", "bin", "obj", "target", "include", "lib"
}

# Logging Configuration
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: str = os.getenv("LOG_FILE", "documind.log")

# Modes
MODES: Dict[str, str] = {
    "LOCAL": "LOCAL",
    "ONLINE": "ONLINE"
}

# Setup Logging
def setup_logging() -> logging.Logger:
    """Setup and return the logger for DocuMind."""
    logger = logging.getLogger("documind")
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (optional)
    if LOG_FILE:
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Global Logger
logger = setup_logging()