from dotenv import load_dotenv
import os 
import ast
import subprocess
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", 
    ".java", ".cpp", ".c", ".h", ".cs", 
    ".go", ".rs", ".php", ".rb", ".swift", 
    ".json", ".yaml", ".yml", ".toml", ".xml",
    ".html", ".css", ".scss",
    ".sql", ".md", ".txt"
}

IGNORE_DIRS = {
    "venv", ".venv", "env", "DocuMind_venv",
    "__pycache__", ".git", ".idea", ".vscode",
    "node_modules", "site-packages", "dist-packages",
    "build", "dist", "bin", "obj", "target", "include", "lib"
}

def _smart_find_file(filename: str) -> str:
    if os.path.exists(filename): return filename
    base = os.path.basename(filename)
    for root, dirs, files in os.walk("./"):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        if base in files: return os.path.join(root, base)
    return None


@tool
def list_project_files():
    """Lists ALL files in the project. Use for 'Show structure' or 'How many files'."""
    file_list = []
    for root, dirs, files in os.walk("./"):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
                file_list.append(os.path.relpath(os.path.join(root, f), "./"))
    return "Files:\n" + "\n".join(sorted(file_list)) if file_list else "No files found."

@tool
def read_specific_file(file_path: str):
    """
    Reads the FULL content of a file. 
    REQUIRED for: Analyzing code logic, checking for hardcoded variables, or reviewing specific functions.
    """
    real_path = _smart_find_file(file_path)
    if not real_path: return f"Error: File '{file_path}' not found."
    try:
        with open(real_path, "r", encoding="utf-8", errors='ignore') as f:
            content = f.read()
            return f"Content of {real_path}:\n{content[:20000]}" 
    except Exception as e: return f"Error: {e}"

@tool
def exact_code_search(keyword: str):
    """Grep search. Use ONLY for finding where a specific string/variable is USED across multiple files."""
    matches = []
    for root, dirs, files in os.walk("./"):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for file in files:
            if os.path.splitext(file)[1].lower() in SUPPORTED_EXTENSIONS:
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', errors='ignore') as f:
                        for i, line in enumerate(f):
                            if keyword in line:
                                matches.append(f"{path}:{i+1}: {line.strip()}")
                                if len(matches) > 20: break
                except: continue
        if len(matches) > 20: break
    return "\n".join(matches) if matches else "No matches found."

@tool
def get_file_structure(file_path: str):
    """Returns ONLY Class and Function names. Use for high-level summaries."""
    real_path = _smart_find_file(file_path)
    if not real_path or not real_path.endswith(".py"): return "Error: Invalid file."
    try:
        with open(real_path, "r") as f: tree = ast.parse(f.read())
    except: return "Error parsing file."
    
    structure = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef): structure.append(f"Function: {node.name}")
        elif isinstance(node, ast.ClassDef): structure.append(f"Class: {node.name}")
    return "\n".join(structure) if structure else "No definitions found."

@tool
def git_history_check(file_path: str):
    """Checks recent git commits for a file."""
    real_path = _smart_find_file(file_path)
    if not real_path: return "File not found."
    try:
        res = subprocess.run(["git", "log", "-n", "3", "--pretty=format:%h %s", "--", real_path], capture_output=True, text=True)
        return res.stdout or "No history."
    except: return "Git error."

@tool
def check_installed_packages(language: str = "python"):
    """Checks pip or npm packages."""
    cmd = ["pip", "list"] if language == "python" else ["npm", "list", "--depth=0"]
    try: return subprocess.run(cmd, capture_output=True, text=True).stdout
    except: return "Error checking packages."

@tool
def write_to_file(file_path: str, content: str):
    """Writes content to a file. Overwrites existing files."""
    if any(x in file_path for x in [".git", "venv"]): return "Forbidden path."
    try:
        with open(file_path, "w", encoding="utf-8") as f: f.write(content)
        return f"Wrote to {file_path}"
    except Exception as e: return f"Error: {e}"


class RAGEngine:
    def __init__(self, index_name:str = None, google_api_key:str = None, pinecone_api_key:str = None):
        if not google_api_key or not pinecone_api_key: raise ValueError("Missing API Keys")
        
        # Setup memory
        self.chat_history = []

        # Setup vector store
        print("Initializing Knowledge Base")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = PineconeVectorStore(index_name=index_name, embedding=self.embeddings, pinecone_api_key=pinecone_api_key)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        @tool
        def semantic_knowledge_search(query: str):
            """
            Use for 'How', 'Why', or conceptual questions.
            """
            return "\n\n".join([d.page_content for d in self.retriever.invoke(query)])

        self.tools = [semantic_knowledge_search, exact_code_search, list_project_files, read_specific_file, get_file_structure, git_history_check, check_installed_packages, write_to_file]

        # Setup LLM & Agent
        print("Initializing Language Model")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", 
                                          temperature=0, 
                                          google_api_key=google_api_key)

        # Smart system prompt
        system_prompt = """You are DocuMind, an expert Senior Software Engineer.
        
        ### MEMORY & CONTEXT
        * **You have memory.** If the user says "check *that* file", refer to the file discussed in the previous turn.
        * **Correct Typos.** If the user asks for "pineconeapi key", assume they mean "PINECONE_API_KEY".

        ### TOOL STRATEGY (CRITICAL)
        1. **Analysis requires READING:** If asked to "check for hardcoded variables", "analyze logic", or "count occurrences", you MUST use `read_specific_file` to see the code first.
        2. **Structure vs. Content:** - `get_file_structure` only gives names (good for summaries).
           - `read_specific_file` gives the actual code (required for debugging).
        3. **Search:** Use `exact_code_search` only for locating strings across the *entire* project.

        ### BEHAVIOR
        * Be concise.
        * If you can't find something, suggest a fix (e.g., "Did you mean 'ingest.py'?").
        """

        self.agent_app = create_agent(
            model=self.llm, 
            tools=self.tools, 
            system_prompt=system_prompt)

    def ask(self, query:str = None):
        print(f"Agent thinking about: {query}")
        
        # Append user message to history
        self.chat_history.append(HumanMessage(content=query))
        
        # Keep history short
        if len(self.chat_history) > 10: 
            self.chat_history = self.chat_history[-10:]

        # Invoke Agent with history
        inputs = {"messages": self.chat_history}
        result = self.agent_app.invoke(inputs)
        
        # Extract & save AI response
        raw_content = result["messages"][-1].content
        if isinstance(raw_content, list):
            final_answer = "".join([block['text'] for block in raw_content if 'text' in block])
        else:
            final_answer = raw_content
            
        self.chat_history.append(AIMessage(content=final_answer))

        return final_answer