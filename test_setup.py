from src.reg import RAGEngine

# Initialize in LOCAL mode
# This uses Ollama (Mistral) + ChromaDB (Local Disk)
print("⏳ Loading Local AI Engine...")
agent = RAGEngine(mode="LOCAL")

print("\n✅ DocuMind Local is ready! (Type 'exit' to quit)")
print("-" * 50)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    # The agent will look at your local files to answer
    response = agent.ask(user_input)
    print(f"🤖 DocuMind: {response}")