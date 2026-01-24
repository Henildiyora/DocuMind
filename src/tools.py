"""
Tools Module for DocuMind

Contains all the LangChain tools used by the RAGEngine for code analysis.
"""

import os
import ast
import subprocess
from typing import List, TYPE_CHECKING
from langchain.tools import tool
from langchain_core.documents import Document

from .config import (
    SUPPORTED_EXTENSIONS,
    IGNORE_DIRS,
    logger
)

if TYPE_CHECKING:
    from .reg import RAGEngine


def _smart_find_file(filename: str) -> str:
    """
    Smart file finder that searches for files by name or partial match.

    Args:
        filename: Name or path of the file to find.

    Returns:
        Full path to the file if found, empty string otherwise.
    """
    if os.path.exists(filename):
        return filename

    base = os.path.basename(filename)
    for root, dirs, files in os.walk("./"):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        if base in files:
            return os.path.join(root, base)
    return ""


@tool
def list_project_files() -> str:
    """
    Lists all supported files in the project.

    Use this tool for 'Show structure' or 'How many files' queries.

    Returns:
        Formatted string of file paths.
    """
    file_list: List[str] = []
    for root, dirs, files in os.walk("./"):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
                file_list.append(os.path.relpath(os.path.join(root, f), "./"))

    if file_list:
        return "Files:\n" + "\n".join(sorted(file_list))
    return "No files found."


@tool
def read_specific_file(file_path: str) -> str:
    """
    Reads the full content of a specific file.

    REQUIRED for: Analyzing code logic, checking for hardcoded variables,
    or reviewing specific functions.

    Args:
        file_path: Path to the file to read.

    Returns:
        File content or error message.
    """
    real_path = _smart_find_file(file_path)
    if not real_path:
        return f"Error: File '{file_path}' not found."

    try:
        with open(real_path, "r", encoding="utf-8", errors='ignore') as f:
            content = f.read()
            return f"Content of {real_path}:\n{content[:20000]}"
    except Exception as e:
        logger.error(f"Error reading file {real_path}: {e}")
        return f"Error: {e}"


@tool
def exact_code_search(keyword: str) -> str:
    """
    Performs grep-style search for a keyword across all files.

    Use ONLY for finding where a specific string/variable is USED
    across multiple files.

    Args:
        keyword: String to search for.

    Returns:
        Matching lines with file and line numbers.
    """
    matches: List[str] = []
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
                                if len(matches) > 20:
                                    break
                except Exception as e:
                    logger.warning(f"Error searching file {path}: {e}")
                    continue
        if len(matches) > 20:
            break

    if matches:
        return "\n".join(matches)
    return "No matches found."


@tool
def get_file_structure(file_path: str) -> str:
    """
    Returns ONLY class and function names from a Python file.

    Use for high-level summaries without reading full content.

    Args:
        file_path: Path to the Python file.

    Returns:
        List of class/function definitions or error message.
    """
    real_path = _smart_find_file(file_path)
    if not real_path or not real_path.endswith(".py"):
        return "Error: Invalid Python file."

    try:
        with open(real_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except Exception as e:
        logger.error(f"Error parsing file {real_path}: {e}")
        return "Error parsing file."

    structure: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            structure.append(f"Function: {node.name}")
        elif isinstance(node, ast.ClassDef):
            structure.append(f"Class: {node.name}")

    if structure:
        return "\n".join(structure)
    return "No definitions found."


@tool
def git_history_check(file_path: str) -> str:
    """
    Checks recent Git commits for a specific file.

    Args:
        file_path: Path to the file to check history for.

    Returns:
        Git log output or error message.
    """
    real_path = _smart_find_file(file_path)
    if not real_path:
        return "File not found."

    try:
        result = subprocess.run(
            ["git", "log", "-n", "3", "--pretty=format:%h %s", "--", real_path],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        return result.stdout or "No history."
    except Exception as e:
        logger.error(f"Git error for {real_path}: {e}")
        return "Git error."


@tool
def check_installed_packages(language: str = "python") -> str:
    """
    Checks installed packages for Python or Node.js.

    Args:
        language: "python" or "javascript" (default: python).

    Returns:
        List of installed packages or error message.
    """
    if language == "python":
        cmd = ["pip", "list"]
    else:
        cmd = ["npm", "list", "--depth=0"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        logger.error(f"Error checking {language} packages: {e}")
        return "Error checking packages."


@tool
def write_to_file(file_path: str, content: str) -> str:
    """
    Writes content to a file. Overwrites existing files.

    Args:
        file_path: Path to the file to write.
        content: Content to write.

    Returns:
        Success message or error.
    """
    if any(x in file_path for x in [".git", "venv"]):
        return "Forbidden path."

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Successfully wrote to {file_path}")
        return f"Wrote to {file_path}"
    except Exception as e:
        logger.error(f"Error writing to {file_path}: {e}")
        return f"Error: {e}"


# Note: semantic_knowledge_search is created dynamically in RAGEngine
# because it needs access to the retriever instance