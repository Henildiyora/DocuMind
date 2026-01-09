import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. Load your API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: GOOGLE_API_KEY not found in .env")
    exit()

# 2. Configure the client
genai.configure(api_key=api_key)

print("🔍 Scanning available Google Models for your API key...\n")
print(f"{'MODEL NAME':<40} | {'CAPABILITIES'}")
print("-" * 60)

try:
    # 3. List all models
    for m in genai.list_models():
        # We only care about models that can generate text (Chat Models)
        # We filter out 'embedContent' models which are for vector databases
        if 'generateContent' in m.supported_generation_methods:
            print(f"{m.name:<40} | {m.supported_generation_methods}")

except Exception as e:
    print(f"❌ Error connecting to Google: {e}")

print("\n✅ Scan Complete. Use one of the names above in your rag.py file.")