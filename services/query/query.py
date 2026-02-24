#!/usr/bin/env python3
"""
Query Pipeline for RAG
Uses PydanticAI with hybrid approach: semantic search + metadata queries.
"""

import os
import json
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
import lancedb
from sentence_transformers import SentenceTransformer
import pyarrow.compute as pc

# Load shared config from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_config = json.loads((_PROJECT_ROOT / "config.json").read_text())
LANCEDB_PATH = _PROJECT_ROOT / _config["lancedb_dir"]
TABLE_NAME = "documents"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# LLM model served by LM Studio (OpenAI-compatible API)
LLM_MODEL = "openai:qwen3-8b"

# Configure LM Studio endpoint (OpenAI-compatible API)
os.environ.setdefault('OPENAI_BASE_URL', 'http://127.0.0.1:1234/v1')
os.environ.setdefault('OPENAI_API_KEY', 'lm-studio')


def check_llm_reachable(timeout: float = 3.0) -> Tuple[bool, str]:
    """Return (True, '') if the LLM server is reachable, else (False, error_message)."""
    base = os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:1234/v1').rstrip('/')
    # OpenAI-compatible servers typically respond on /models
    url = f"{base}/models"
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"})
        urllib.request.urlopen(req, timeout=timeout)
        return True, ""
    except urllib.error.HTTPError:
        # Server responded (e.g. 401/403) -> reachable
        return True, ""
    except urllib.error.URLError as e:
        if "Connection refused" in str(e.reason) or "refused" in str(e.reason).lower():
            return False, f"LLM server not reachable at {base}. Is LM Studio running with a model loaded?"
        if "timed out" in str(e.reason).lower():
            return False, f"LLM server at {base} did not respond within {timeout}s."
        return False, f"LLM server error: {e.reason}"
    except OSError as e:
        return False, f"Cannot reach LLM server: {e}"


# --- Response models ---

class FileInfo(BaseModel):
    """Structured file information."""
    file_name: str
    file_type: str
    source_path: str
    first_paragraph: str
    timestamp: str
    total_chunks: Optional[int] = None


# --- Database layer ---

class DocumentDatabase:
    """Handles all database operations for the RAG agent."""

    def __init__(self):
        self.db_path = LANCEDB_PATH
        self.table_name = TABLE_NAME

        # Initialize embedding model for semantic search
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        print("✓ Model loaded")

        # Connect to LanceDB
        self.db = lancedb.connect(str(self.db_path))
        try:
            self.table = self.db.open_table(self.table_name)
            print(f"✓ Connected to database: {self.table_name}")
        except Exception:
            print(f"✗ Database table '{self.table_name}' not found. Run ingest first.")
            self.table = None

    def refresh_table(self) -> None:
        """Re-open the table so queries see the latest data (e.g. after pipeline ingest)."""
        if self.db is None:
            return
        try:
            self.table = self.db.open_table(self.table_name)
        except Exception:
            self.table = None

    def semantic_search(
        self, query: str, limit: int = 5, min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search for semantically similar chunks.
        min_score: if set, only return results with similarity >= min_score (0-1).
        Uses cosine distance so _distance = 1 - similarity.
        """
        if self.table is None:
            return []
        query_vector = self.embedder.encode(query).tolist()
        q = (
            self.table.search(query_vector)
            .distance_type("cosine")
            .limit(limit)
        )
        if min_score is not None:
            q = q.where(f"_distance < {1.0 - min_score}")
        return q.to_list()

    def count_files(self) -> int:
        """Count total unique files in database."""
        if self.table is None:
            return 0
        table = self.table.to_arrow()
        return len(set(table['md_path'].to_pylist()))

    def find_file_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Find files matching a name pattern (substring, case-insensitive)."""
        if self.table is None:
            return []

        table = self.table.to_arrow()
        mask = pc.match_substring(table["file_name"], name, ignore_case=True)
        filtered = pc.filter(table, mask)

        # Deduplicate by file_name, one dict per file (same shape as before)
        seen: set = set()
        matches = []
        for i in range(filtered.num_rows):
            file_name = filtered["file_name"][i].as_py()
            if file_name in seen:
                continue
            seen.add(file_name)
            matches.append({
                "file_name": file_name,
                "file_type": filtered["file_type"][i].as_py(),
                "source_path": filtered["source_path"][i].as_py(),
                "first_paragraph": filtered["first_paragraph"][i].as_py(),
                "timestamp": filtered["indexed_at"][i].as_py(),
                "total_chunks": filtered["total_chunks"][i].as_py(),
            })
        return matches

    def list_files_by_date(self, days_ago: int = 1) -> List[Dict[str, Any]]:
        """List files created within last N days."""
        if self.table is None:
            return []

        table = self.table.to_arrow()
        cutoff_date = (datetime.now() - timedelta(days=days_ago)).isoformat()

        files = {}
        for idx in range(len(table)):
            timestamp = table['indexed_at'][idx].as_py()
            if timestamp >= cutoff_date:
                file_name = table['file_name'][idx].as_py()
                if file_name not in files:
                    files[file_name] = {
                        'file_name': file_name,
                        'file_type': table['file_type'][idx].as_py(),
                        'source_path': table['source_path'][idx].as_py(),
                        'first_paragraph': table['first_paragraph'][idx].as_py(),
                        'timestamp': timestamp,
                        'total_chunks': table['total_chunks'][idx].as_py()
                    }

        return sorted(files.values(), key=lambda x: x['timestamp'], reverse=True)

    def get_files_by_type(self, file_type: str) -> List[Dict[str, Any]]:
        """Get files of a specific type (pdf, txt, image_ocr, etc)."""
        if self.table is None:
            return []

        table = self.table.to_arrow()
        type_lower = file_type.lower()

        files = {}
        for idx in range(len(table)):
            ftype = table['file_type'][idx].as_py()
            if type_lower in ftype.lower():
                file_name = table['file_name'][idx].as_py()
                if file_name not in files:
                    files[file_name] = {
                        'file_name': file_name,
                        'file_type': ftype,
                        'source_path': table['source_path'][idx].as_py(),
                        'first_paragraph': table['first_paragraph'][idx].as_py(),
                        'timestamp': table['indexed_at'][idx].as_py()
                    }

        return list(files.values())

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        if self.table is None:
            return {"error": "No database connected"}

        table = self.table.to_arrow()

        total_chunks = len(table)
        unique_files = len(set(table['md_path'].to_pylist()))
        file_types = {}

        for idx in range(len(table)):
            ftype = table['file_type'][idx].as_py()
            file_types[ftype] = file_types.get(ftype, 0) + 1

        return {
            'total_chunks': total_chunks,
            'unique_files': unique_files,
            'file_types': file_types,
        }

    def list_all_files(self) -> List[Dict[str, Any]]:
        """List all files in the database with their metadata."""
        if self.table is None:
            return []

        table = self.table.to_arrow()

        files = {}
        for idx in range(len(table)):
            md_path = table['md_path'][idx].as_py()
            if md_path not in files:
                files[md_path] = {
                    'file_name': table['file_name'][idx].as_py(),
                    'file_type': table['file_type'][idx].as_py(),
                    'source_path': table['source_path'][idx].as_py(),
                    'first_paragraph': table['first_paragraph'][idx].as_py(),
                    'timestamp': table['indexed_at'][idx].as_py(),
                    'total_chunks': table['total_chunks'][idx].as_py()
                }

        return list(files.values())

    def get_file_content(self, file_name: str) -> List[Dict[str, Any]]:
        """Get all chunks from a specific file, ordered by chunk_index."""
        if self.table is None:
            return []

        table = self.table.to_arrow()
        chunks = []

        for idx in range(len(table)):
            if table['file_name'][idx].as_py() == file_name:
                chunks.append({
                    'chunk_index': table['chunk_index'][idx].as_py(),
                    'text': table['text'][idx].as_py(),
                    'file_name': file_name,
                    'total_chunks': table['total_chunks'][idx].as_py()
                })

        chunks.sort(key=lambda x: x['chunk_index'])
        return chunks

    def get_all_files_content(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get content of all files in database."""
        if self.table is None:
            return {}

        files = self.list_all_files()
        return {f['file_name']: self.get_file_content(f['file_name']) for f in files}


# --- Formatting helpers ---

def _format_file_tree(files: List[FileInfo]) -> str:
    """Format files as a tree structure."""
    if not files:
        return "No files found."

    output = ["Files:"]
    for i, file in enumerate(files, 1):
        is_last = i == len(files)
        prefix = "└── " if is_last else "├── "

        output.append(f"{prefix}{file.file_name}")

        indent = "    " if is_last else "│   "
        output.append(f"{indent}├─ Type: {file.file_type}")
        output.append(f"{indent}├─ Path: {file.source_path}")
        if file.first_paragraph:
            preview = (file.first_paragraph[:120] + "…") if len(file.first_paragraph) > 120 else file.first_paragraph
            output.append(f"{indent}├─ First paragraph: {preview}")
        if file.total_chunks:
            output.append(f"{indent}├─ Chunks: {file.total_chunks}")
        output.append(f"{indent}└─ Modified: {file.timestamp}")

        if not is_last:
            output.append("│")

    return "\n".join(output)


def _format_search_results(results: List[Dict[str, Any]]) -> str:
    """Format search results as a tree structure."""
    if not results:
        return "No results found."

    output = ["Search Results:"]
    for i, result in enumerate(results, 1):
        is_last = i == len(results)
        prefix = "└── " if is_last else "├── "

        score = result.get('_distance', 0)
        relevance = f"{(1 - score) * 100:.1f}%" if score < 1 else "N/A"

        output.append(f"{prefix}Result {i} (Relevance: {relevance})")

        indent = "    " if is_last else "│   "
        output.append(f"{indent}├─ File: {result.get('file_name', 'Unknown')}")
        output.append(f"{indent}├─ Chunk: {result.get('chunk_index', 0) + 1}/{result.get('total_chunks', 1)}")
        output.append(f"{indent}└─ Text:")

        text = result.get('text', '').strip()
        text_lines = text.split('\n')
        text_indent = f"{indent}    "
        for line in text_lines[:5]:
            output.append(f"{text_indent}{line}")
        if len(text_lines) > 5:
            output.append(f"{text_indent}...")

        if not is_last:
            output.append("│")

    return "\n".join(output)


def _format_stats(stats: Dict[str, Any]) -> str:
    """Format database statistics as a tree structure."""
    output = []
    output.append("Database Statistics:")
    output.append(f"├── Total Chunks: {stats.get('total_chunks', 0)}")
    output.append(f"├── Unique Files: {stats.get('unique_files', 0)}")

    if 'file_types' in stats:
        output.append("└── File Types:")
        types = stats['file_types']
        for i, (ftype, count) in enumerate(types.items(), 1):
            is_last_type = i == len(types)
            type_prefix = "    └── " if is_last_type else "    ├── "
            output.append(f"{type_prefix}{ftype}: {count}")

    return "\n".join(output)


# --- Initialize database and agent at module level ---

db = DocumentDatabase()

agent = Agent(
    LLM_MODEL,
    system_prompt="""You are a helpful document assistant. You have access to a vector database containing document content.

TOOL USAGE:
- For content questions ("what's my address?", "find X"): use semantic_search()
- For file listings: use list_all_files()
- For summaries: use summarize_all_files() or get_file_content()
- For metadata queries: use find_file_by_name(), get_files_by_type(), etc.

OUTPUT RULES:
- When tools return formatted text, output it exactly as-is
- Preserve all Unicode characters and spacing
- Never translate or explain filenames
- Never make up information
- For summaries, READ content and generate actual summaries

NEVER try to retrieve files by typing filenames with special characters.""",
)


# --- Agent tools ---

@agent.tool
def semantic_search(ctx: RunContext, query: str, limit: int = 10) -> str:
    """Search for content within documents using semantic similarity.

    Args:
        query: The search query (e.g., "address", "legal terms", "health information")
        limit: Maximum results, default 10

    Returns:
        Pre-formatted string with search results including filename and content
    """
    results = db.semantic_search(query, limit)
    if not results:
        return "No relevant results found."
    return _format_search_results(results)


@agent.tool
def count_files(ctx: RunContext) -> int:
    """Count the total number of unique files in the database."""
    return db.count_files()


@agent.tool
def find_file_by_name(ctx: RunContext, name: str) -> str:
    """Find files by name (partial match, case-insensitive).

    Args:
        name: File name or partial name to search for
    """
    files = db.find_file_by_name(name)
    if not files:
        return "No files found matching that name."
    return _format_file_tree([FileInfo(**f) for f in files])


@agent.tool
def list_files_by_date(ctx: RunContext, days_ago: int = 1) -> str:
    """List files created within the last N days.

    Args:
        days_ago: Number of days to look back (default 1)
    """
    files = db.list_files_by_date(days_ago)
    if not files:
        return f"No files found from the last {days_ago} day(s)."
    return _format_file_tree([FileInfo(**f) for f in files])


@agent.tool
def get_files_by_type(ctx: RunContext, file_type: str) -> str:
    """Get files of a specific type.

    Args:
        file_type: File type (pdf, txt, pdf_ocr, image_ocr, html, docx, etc)
    """
    files = db.get_files_by_type(file_type)
    if not files:
        return f"No files found of type '{file_type}'."
    return _format_file_tree([FileInfo(**f) for f in files])


@agent.tool
def get_database_stats(ctx: RunContext) -> str:
    """Get comprehensive database statistics (counts only, not file names)."""
    stats = db.get_database_stats()
    if 'error' in stats:
        return stats['error']
    return _format_stats(stats)


@agent.tool
def list_all_files(ctx: RunContext) -> str:
    """List ALL files in the database with complete metadata.
    Use when user asks to "list files", "show all files", "what files do I have", etc.
    """
    files = db.list_all_files()
    if not files:
        return "No files found in database."
    return _format_file_tree([FileInfo(**f) for f in files])


@agent.tool
def get_file_content(ctx: RunContext, file_name: str) -> str:
    """Get the complete text content of a specific file.
    Use this to read file contents, summarize documents, or answer questions about specific files.

    Args:
        file_name: Exact name of the file (use list_all_files first to get exact names)
    """
    chunks = db.get_file_content(file_name)
    if not chunks:
        return f"File '{file_name}' not found. Use list_all_files to see available files."

    output = [f"Content of '{file_name}':", f"Total chunks: {chunks[0]['total_chunks']}\n"]
    for chunk in chunks:
        output.append(f"--- Chunk {chunk['chunk_index'] + 1}/{chunk['total_chunks']} ---")
        output.append(chunk['text'])
        output.append("")

    return "\n".join(output)


@agent.tool
def summarize_all_files(ctx: RunContext) -> str:
    """Get content of ALL files in the database for summarization.
    Use when user asks to "summarize each file", "give me a summary of all files", etc.
    Returns file names with content (up to 1000 chars per file) for you to summarize.
    """
    all_content = db.get_all_files_content()
    if not all_content:
        return "No files found in database."

    output = ["Files to Summarize:", ""]

    files = list(all_content.items())
    for i, (file_name, chunks) in enumerate(files, 1):
        is_last = i == len(files)
        prefix = "└── " if is_last else "├── "

        output.append(f"{prefix}FILE: {file_name}")
        indent = "    " if is_last else "│   "

        if chunks:
            full_text = "\n".join(chunk['text'] for chunk in chunks)
            content = full_text[:1000].strip()
            if len(full_text) > 1000:
                content += "..."
            output.append(f"{indent}CONTENT:")
            for line in content.split('\n'):
                output.append(f"{indent}  {line}")
        else:
            output.append(f"{indent}(No content)")

        if not is_last:
            output.append("│")
        output.append("")

    return "\n".join(output)


# --- CLI ---

def main():
    """Interactive CLI for querying documents."""
    parser = argparse.ArgumentParser(description="Query your document knowledge base")
    parser.add_argument("query", nargs="?", help="Your question")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if db.table is None:
        print("No database found. Run ingest first to index your documents.")
        return

    ok, err = check_llm_reachable()
    if not ok:
        print(err)
        return

    print("=" * 60)
    print("RAG Query Assistant (LM Studio + PydanticAI)")
    print("=" * 60)
    print(f"Database: {db.count_files()} files indexed")
    print()

    if args.query:
        print(f"Q: {args.query}\n")
        result = agent.run_sync(args.query)
        print(f"A: {result.output}\n")

    elif args.interactive:
        print("Interactive mode. Type 'exit' or 'quit' to end.\n")

        while True:
            try:
                query = input("Q: ").strip()

                if query.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break

                if not query:
                    continue

                print()
                result = agent.run_sync(query)
                print(f"A: {result.output}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python query.py 'How many files do I have?'")
        print("  python query.py 'What does document X say about topic Y?'")
        print("  python query.py --interactive")


if __name__ == "__main__":
    main()
