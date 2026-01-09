from dotenv import load_dotenv
import os 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class RAGEngine:
    '''
    A Retrieval-Augmented Generation (RAG) engine using Google Generative AI and Pinecone.
    '''
    def __init__(self,index_name:str = None ,google_api_key:str = None, pinecone_api_key:str = None):

        self.GOOGLE_API_KEY = google_api_key
        self.PINECONE_API_KEY = pinecone_api_key
        self.INDEX_NAME = index_name
        if not self.GOOGLE_API_KEY or not self.PINECONE_API_KEY:
            raise ValueError("API keys not found. Please set them in the .env file.")
        
        # We use the local HuggingFace model to convert the user's question into numbers
        print("Initializing Embeddings (Local)")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Connect to the Pinecone vector store
        print("Connecting to Pinecone Vector Store")
        self.vector_store = PineconeVectorStore(
            index_name=self.INDEX_NAME,
            embedding=self.embeddings,
            pinecone_api_key=self.PINECONE_API_KEY
        )

        # Setup the retriever
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        # Setup the LLM
        print("Initializing Google Generative AI LLM")
        self.llm = ChatGoogleGenerativeAI(
            model = "gemini-2.5-flash",
            temperature = 0,
            google_api_key = self.GOOGLE_API_KEY
        )

        # setup the prompt template 
        self.template = """
        You are an intelligent assistant for a company. 
        Answer the question based ONLY on the following context. 
        If the answer is not in the context, say "I don't know based on the available documents.

        Context = {context}

        Question = {question}
        """

        self.prompt = ChatPromptTemplate.from_template(self.template)

        # Build a RAG chain
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self,query:str = None):
        """
        Public method to ask a question.
        """
        print(f"Asking question: {query}")

        # Invoke the chain
        # 1. Retriever finds docs
        # 2. Prompt fills in context + question
        # 3. LLM generates answer
        response = self.chain.invoke(query)
        
        return response
