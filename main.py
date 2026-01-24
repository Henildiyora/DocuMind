#!/usr/bin/env python3
"""
DocuMind Main Entry Point

This script provides the command-line interface for DocuMind,
an AI-powered code analysis and documentation agent.
"""

import os
import sys
import argparse
from typing import NoReturn
from src.reg import RAGEngine
from src.ingest import IngestionPipeline
from src.config import (
    PINECONE_API_KEY,
    GOOGLE_API_KEY,
    INDEX_NAME,
    EMBEDDING_DIMENSION,
    logger
)


def run_ingestion(target_path: str) -> None:
    """
    Runs the ingestion pipeline for the specified target path.

    Args:
        target_path: Path to file or directory to ingest.

    Raises:
        SystemExit: If ingestion fails critically.
    """
    logger.info(f"Starting ingestion for: {target_path}")
    try:
        pipeline = IngestionPipeline(
            pinecone_api_key=PINECONE_API_KEY,
            index_name=INDEX_NAME,
            embedding_dimension=EMBEDDING_DIMENSION
        )
        pipeline.run(directory_path=target_path)
        logger.info("Ingestion completed successfully")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)


def start_chat(bot: RAGEngine) -> NoReturn:
    """
    Starts the interactive chat loop with the DocuMind agent.

    Args:
        bot: Initialized RAGEngine instance.

    Raises:
        SystemExit: On keyboard interrupt or critical error.
    """
    print("\n" + "="*50)
    print(" DocuMind Agent Ready")
    print("   - Type your question to chat.")
    print("   - Type 'exit' or 'quit' to stop.")
    print("   - Type 'clear' to clear the screen.")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("\nExiting DocuMind.")
                break
            elif user_input.lower() == "clear":
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            elif not user_input:
                continue

            logger.info(f"Processing query: {user_input[:50]}...")
            print("Thinking...")
            response = bot.ask(user_input)

            print("\nDocuMind:")
            print(response)
            print("-" * 50)

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, exiting")
            print("\n\nGoodbye")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Chat error: {e}")
            print(f"\nError: {e}")


def main() -> NoReturn:
    """
    Main entry point for DocuMind CLI.

    Parses arguments, optionally runs ingestion, initializes agent,
    and starts chat loop.
    """
    parser = argparse.ArgumentParser(
        description="DocuMind AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Ingest current dir and chat
  python main.py --target /path/to/code   # Ingest specific path
  python main.py --no-ingest              # Skip ingestion, just chat
        """
    )
    parser.add_argument(
        "--target",
        type=str,
        default=".",
        help="Path to file or folder to ingest (default: current directory)"
    )
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Skip ingestion and just start chat mode"
    )
    args = parser.parse_args()

    if args.no_ingest:
        logger.info("Skipping ingestion as per --no-ingest flag")
    else:
        run_ingestion(args.target)

    try:
        bot = RAGEngine(
            index_name=INDEX_NAME,
            google_api_key=GOOGLE_API_KEY,
            pinecone_api_key=PINECONE_API_KEY
        )
        start_chat(bot)
    except Exception as e:
        logger.critical(f"Critical system error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()