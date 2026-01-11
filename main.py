import os
import sys
import argparse
from src.reg import RAGEngine
from src.ingest import IngestionPipeline

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "documind-code-docs-index"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def run_ingestion(target_path):
    """Runs the ingestion pipeline for the specific target."""
    print(f"\n[AUTO-INGEST] Loading code from: {target_path}")
    try:
        pipeline = IngestionPipeline(
            pinecone_api_key=PINECONE_API_KEY, 
            index_name=INDEX_NAME,
            embedding_dimension=384
        )
        pipeline.run(directory_path=target_path)
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {e}")
        exit(1)


def main():

    # Parse Arguments
    parser = argparse.ArgumentParser(description="DocuMind AI Agent")
    parser.add_argument("--target",type=str,help="Path to file or folder to ingest (default: current dir)",default=".")
    parser.add_argument("--no-ingest",action="store_true", help="Skip ingestion and just chat")
    args = parser.parse_args()

    if args.no_ingest:
        print("\n[INFO] Skipping ingestion as per --no-ingest flag.")
    else:
        run_ingestion(args.target)

    try:
        bot = RAGEngine(
            index_name=INDEX_NAME,
            google_api_key=GOOGLE_API_KEY,
            pinecone_api_key=PINECONE_API_KEY
        )
        print("\n" + "="*50)
        print(" DocuMind Agent Ready")
        print("   - Type your question to chat.")
        print("   - Type 'exit' or 'quit' to stop.")
        print("   - Type 'clear' to clear the screen.")
        print("="*50 + "\n")

        # 4. Continuous Chat Loop
        while True:
            try:
                # Get User Input
                user_input = input("\n You: ").strip()

                # Handle Commands
                if user_input.lower() in ["exit", "quit"]:
                    print("\n Exiting DocuMind.")
                    break
                
                if user_input.lower() == "clear":
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue

                if not user_input:
                    continue

                # Get Response
                print("Thinking")
                response = bot.ask(user_input)
                
                # Print Response (Markdown formatting is handled by the Agent)
                print("\n DocuMind:")
                print(response)
                print("-" * 50)

            except KeyboardInterrupt:
                print("\n\n Goodbye")
                sys.exit(0)
            except Exception as e:
                print(f"\n Error: {e}")

    except Exception as e:
        print(f"Critical System Error: {e}")

if __name__ == "__main__":
    main()