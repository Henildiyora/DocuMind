import os
from dotenv import load_dotenv

# Load secrets from .env file
load_dotenv()

google_key = os.getenv("GOOGLE_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

if google_key and pinecone_key:
    print("Environment variables are loaded securely.")
else:
    print("Keys not found. Check your .env file.")