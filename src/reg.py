"""
RAGEngine Module

Contains the main RAGEngine class that orchestrates the DocuMind agent,
including vector stores, LLMs, and tool execution.
"""

from typing import List, Optional, Any, Dict
from langchain.agents import create_agent
from langchain.tools import BaseTool, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

from .config import (
    MODES,
    DEFAULT_LLM_MODEL,
    LOCAL_LLM_MODEL,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    LOCAL_VECTOR_DIR,
    COLLECTION_NAME,
    MAX_RETRIEVAL_RESULTS,
    MAX_CHAT_HISTORY,
    logger
)
from .tools import (
    exact_code_search,
    list_project_files,
    read_specific_file,
    get_file_structure,
    git_history_check,
    check_installed_packages,
    write_to_file
)
from .prompts import SYSTEM_PROMPT


class RAGEngine:
    """
    Retrieval-Augmented Generation Engine for DocuMind.

    Supports both online (Pinecone + Gemini) and local (ChromaDB + Ollama) modes.
    """

    def __init__(
        self,
        mode: str = "LOCAL",
        index_name: Optional[str] = None,
        google_api_key: Optional[str] = None,
        pinecone_api_key: Optional[str] = None
    ) -> None:
        """
        Initialize the RAG Engine.

        Args:
            mode: "LOCAL" or "ONLINE".
            index_name: Name of the vector index.
            google_api_key: API key for Google Gemini.
            pinecone_api_key: API key for Pinecone.
        """
        self.mode = mode.upper()
        if self.mode not in MODES:
            raise ValueError(f"Invalid mode: {mode}. Must be 'LOCAL' or 'ONLINE'")

        self.chat_history: List[Any] = []
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        logger.info(f"Initializing RAGEngine in {self.mode} mode")

        # Setup vector store and LLM based on mode
        if self.mode == "ONLINE":
            self._setup_online(index_name, google_api_key, pinecone_api_key)
        else:
            self._setup_local()

        # Setup tools
        self.tools: List[BaseTool] = [
            self._create_semantic_search_tool(),
            exact_code_search,
            list_project_files,
            read_specific_file,
            get_file_structure,
            git_history_check,
            check_installed_packages,
            write_to_file
        ]

        # Create agent
        self.agent_app = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT
        )

        logger.info("RAGEngine initialized successfully")

    def _create_semantic_search_tool(self) -> BaseTool:
        """Create the semantic knowledge search tool with retriever access."""
        @tool
        def semantic_knowledge_search(query: str) -> str:
            """
            Use for 'How', 'Why', or conceptual questions.
            Searches the knowledge base for relevant information.
            """
            try:
                docs = self.retriever.invoke(query)
                return "\n\n".join([d.page_content for d in docs])
            except Exception as e:
                logger.error(f"Semantic search failed: {e}")
                return f"Error in semantic search: {e}"

        return semantic_knowledge_search

    def _setup_online(
        self,
        index_name: Optional[str],
        google_api_key: Optional[str],
        pinecone_api_key: Optional[str]
    ) -> None:
        """Setup online mode with Pinecone and Gemini."""
        if not google_api_key or not pinecone_api_key:
            raise ValueError("Missing API keys for ONLINE mode")

        self.vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=self.embeddings,
            pinecone_api_key=pinecone_api_key
        )
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": MAX_RETRIEVAL_RESULTS}
        )

        logger.info("Setting up Google Gemini LLM")
        self.llm = ChatGoogleGenerativeAI(
            model=DEFAULT_LLM_MODEL,
            temperature=0,
            google_api_key=google_api_key
        )

    def _setup_local(self) -> None:
        """Setup local mode with ChromaDB and Ollama."""
        logger.info("Setting up local ChromaDB vector store")
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=LOCAL_VECTOR_DIR
        )
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": MAX_RETRIEVAL_RESULTS}
        )

        logger.info("Setting up Ollama LLM")
        try:
            self.llm = ChatOllama(
                model=LOCAL_LLM_MODEL,
                temperature=0,
                base_url=OLLAMA_BASE_URL
            )
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            raise

    def ask(self, query: str) -> str:
        """
        Process a user query and return the agent's response.

        Args:
            query: User's question or command.

        Returns:
            Agent's response as a string.
        """
        logger.info(f"Processing query: {query[:50]}...")

        # Add user message to history
        self.chat_history.append(HumanMessage(content=query))

        # Trim history to prevent bloat
        if len(self.chat_history) > MAX_CHAT_HISTORY:
            self.chat_history = self.chat_history[-MAX_CHAT_HISTORY:]

        # Invoke agent
        inputs: Dict[str, Any] = {"messages": self.chat_history}
        try:
            result = self.agent_app.invoke(inputs)
        except Exception as e:
            logger.error(f"Agent invocation failed: {e}")
            return f"Error processing query: {e}"

        # Extract response
        messages = result.get("messages", [])
        if not messages:
            logger.error("No messages in agent response")
            return "Error: No response from agent"

        last_message = messages[-1]
        raw_content = last_message.content
        if isinstance(raw_content, list):
            final_answer = "".join(
                [block.get('text', '') for block in raw_content if 'text' in block]
            )
        else:
            final_answer = str(raw_content)

        # Add to history
        self.chat_history.append(AIMessage(content=final_answer))

        logger.info("Query processed successfully")
        return final_answer