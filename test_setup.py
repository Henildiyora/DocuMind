#!/usr/bin/env python3
"""
Test Setup Script for DocuMind

Quick test harness for the local RAGEngine mode.
"""

from src.reg import RAGEngine

def main() -> None:
    """Run the test chat loop."""
    print("⏳ Loading Local AI Engine...")
    agent = RAGEngine(mode="LOCAL")

    print("\n✅ DocuMind Local is ready! (Type 'exit' to quit)")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        # The agent will look at your local files to answer
        response = agent.ask(user_input)
        print(f"🤖 DocuMind: {response}")

if __name__ == "__main__":
    main()