from src.reg import RAGEngine
import os

def main():

    print(" Initializing DocuMind RAG Engine")

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = "documind-index"
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Initialize the Engine (connects to DB and LLM)
    engine = RAGEngine(index_name=INDEX_NAME, pinecone_api_key=PINECONE_API_KEY,google_api_key=GOOGLE_API_KEY)
    
    print(" System Ready | Type 'exit' to stop.")
    print("-" * 50)

    while True:
        user_query = input("You: ")

        if user_query.lower() in ['exit', 'quit']:
            print("Exiting DocuMind.")
            break

        try:

            # Get the answer 
            answer = engine.ask(query=user_query)

            print("\n🤖 DocuMind Answer:")
            print(answer)
            print("-" * 50)

        except Exception as e:
            print(f" An error occurred: {e}")


if __name__ == "__main__":
    main()