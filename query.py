#!/usr/bin/env python3
"""
Query Pipeline for RAG
Uses PydanticAI with hybrid approach: semantic search + metadata queries.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import os
import json
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
import lancedb
from sentence_transformers import SentenceTransformer
import pyarrow.compute as pc

# Response models for structured output
class FileInfo(BaseModel):
    """Structured file information."""
    file_name: str
    file_type: str
    source_path: str
    tags: List[str]
    timestamp: str
    total_chunks: Optional[int] = None

class QueryResponse(BaseModel):
    """Structured response for all queries."""
    answer: str = Field(description="Natural language answer to the user's question")
    files: Optional[List[FileInfo]] = Field(default=None, description="List of files if listing files")
    search_results: Optional[List[Dict[str, Any]]] = Field(default=None, description="Search results with text chunks")
    stats: Optional[Dict[str, Any]] = Field(default=None, description="Database statistics")

# Configuration
LANCEDB_PATH = Path("lancedb")
TABLE_NAME = "documents"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Configure LM Studio endpoint (OpenAI-compatible API)
os.environ.setdefault('OPENAI_BASE_URL', 'http://127.0.0.1:1234/v1')
os.environ.setdefault('OPENAI_API_KEY', 'lm-studio')  # LM Studio doesn't need real key


class DocumentDatabase:
    """Handles all database operations for the RAG agent."""
    
    def __init__(self):
        self.db_path = LANCEDB_PATH
        self.table_name = TABLE_NAME
        
        # Initialize embedding model for semantic search
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✓ Model loaded")
        
        # Connect to LanceDB
        self.db = lancedb.connect(str(self.db_path))
        try:
            self.table = self.db.open_table(self.table_name)
            print(f"✓ Connected to database: {self.table_name}")
        except:
            print(f"✗ Database table '{self.table_name}' not found. Run 'make ingest' first.")
            self.table = None
    
    def semantic_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for semantically similar chunks."""
        if self.table is None:
            return []
        
        # Generate query embedding
        query_vector = self.embedder.encode(query).tolist()
        
        # Search and return results
        results = self.table.search(query_vector).limit(limit).to_list()
        return results
    
    def count_files(self) -> int:
        """Count total unique files in database."""
        if self.table is None:
            return 0
        
        table = self.table.to_arrow()
        unique_files = len(set(table['md_path'].to_pylist()))
        return unique_files
    
    def find_file_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Find files matching a name pattern."""
        if self.table is None:
            return []
        
        table = self.table.to_arrow()
        
        # Case-insensitive search in file_name
        name_lower = name.lower()
        matches = []
        
        for idx in range(len(table)):
            file_name = table['file_name'][idx].as_py()
            if name_lower in file_name.lower():
                # Get unique file info
                file_info = {
                    'file_name': file_name,
                    'file_type': table['file_type'][idx].as_py(),
                    'source_path': table['source_path'][idx].as_py(),
                    'tags': table['tags'][idx].as_py(),
                    'timestamp': table['timestamp'][idx].as_py(),
                    'total_chunks': table['total_chunks'][idx].as_py()
                }
                # Add only unique files
                if not any(m['file_name'] == file_info['file_name'] for m in matches):
                    matches.append(file_info)
        
        return matches
    
    def list_files_by_date(self, days_ago: int = 1) -> List[Dict[str, Any]]:
        """List files created within last N days."""
        if self.table is None:
            return []
        
        table = self.table.to_arrow()
        cutoff_date = (datetime.now() - timedelta(days=days_ago)).isoformat()
        
        files = {}
        for idx in range(len(table)):
            timestamp = table['timestamp'][idx].as_py()
            if timestamp >= cutoff_date:
                file_name = table['file_name'][idx].as_py()
                if file_name not in files:
                    files[file_name] = {
                        'file_name': file_name,
                        'file_type': table['file_type'][idx].as_py(),
                        'source_path': table['source_path'][idx].as_py(),
                        'tags': table['tags'][idx].as_py(),
                        'timestamp': timestamp,
                        'total_chunks': table['total_chunks'][idx].as_py()
                    }
        
        return sorted(files.values(), key=lambda x: x['timestamp'], reverse=True)
    
    def get_files_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Get files with a specific tag (folder)."""
        if self.table is None:
            return []
        
        table = self.table.to_arrow()
        tag_lower = tag.lower()
        
        files = {}
        for idx in range(len(table)):
            tags = table['tags'][idx].as_py()
            if any(tag_lower in t.lower() for t in tags):
                file_name = table['file_name'][idx].as_py()
                if file_name not in files:
                    files[file_name] = {
                        'file_name': file_name,
                        'file_type': table['file_type'][idx].as_py(),
                        'source_path': table['source_path'][idx].as_py(),
                        'tags': tags,
                        'timestamp': table['timestamp'][idx].as_py()
                    }
        
        return list(files.values())
    
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
                        'tags': table['tags'][idx].as_py(),
                        'timestamp': table['timestamp'][idx].as_py()
                    }
        
        return list(files.values())
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        if self.table is None:
            return {"error": "No database connected"}
        
        table = self.table.to_arrow()
        
        # Collect stats
        total_chunks = len(table)
        unique_files = len(set(table['md_path'].to_pylist()))
        file_types = {}
        tags_count = {}
        
        for idx in range(len(table)):
            # Count file types
            ftype = table['file_type'][idx].as_py()
            file_types[ftype] = file_types.get(ftype, 0) + 1
            
            # Count tags (only count once per file)
            tags = table['tags'][idx].as_py()
            for tag in tags:
                tags_count[tag] = tags_count.get(tag, 0) + 1
        
        return {
            'total_chunks': total_chunks,
            'unique_files': unique_files,
            'file_types': file_types,
            'tags': tags_count
        }
    
    def list_all_files(self) -> List[Dict[str, Any]]:
        """List all files in the database with their metadata."""
        if self.table is None:
            return []
        
        table = self.table.to_arrow()
        
        # Get unique files
        files = {}
        for idx in range(len(table)):
            md_path = table['md_path'][idx].as_py()
            if md_path not in files:
                files[md_path] = {
                    'file_name': table['file_name'][idx].as_py(),
                    'file_type': table['file_type'][idx].as_py(),
                    'source_path': table['source_path'][idx].as_py(),
                    'tags': table['tags'][idx].as_py(),
                    'timestamp': table['timestamp'][idx].as_py(),
                    'total_chunks': table['total_chunks'][idx].as_py()
                }
        
        return list(files.values())
    
    def get_file_content(self, file_name: str) -> List[Dict[str, Any]]:
        """Get all chunks from a specific file.
        
        Args:
            file_name: Name of the file to retrieve
        
        Returns:
            List of chunks with text content, ordered by chunk_index
        """
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
        
        # Sort by chunk_index
        chunks.sort(key=lambda x: x['chunk_index'])
        return chunks
    
    def get_all_files_content(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get content of all files in database.
        
        Returns:
            Dictionary mapping file_name to list of chunks
        """
        if self.table is None:
            return {}
        
        files = self.list_all_files()
        result = {}
        
        for file in files:
            file_name = file['file_name']
            result[file_name] = self.get_file_content(file_name)
        
        return result


# Initialize database
db = DocumentDatabase()


# Define agent with LM Studio (qwen3-14b)
agent = Agent(
    'openai:qwen3-14b',  # Using LM Studio's model
    system_prompt="""You are a helpful document assistant. You have access to a vector database containing document content.

TOOL USAGE:
- For content questions ("what's my address?", "find X"): use semantic_search()
- For file listings: use list_all_files()
- For summaries: use summarize_all_files() or get_file_content()
- For metadata queries: use find_file_by_name(), get_files_by_tag(), etc.

OUTPUT RULES:
- When tools return formatted text (with 📁 or 🔍), output it exactly as-is
- Preserve all Unicode characters and spacing
- Never translate or explain filenames
4. Never make up information
5. For summaries, READ content and generate actual summaries

NEVER try to retrieve files by typing filenames with special characters.""",
)


# Register tools
@agent.tool
def semantic_search(ctx: RunContext, query: str, limit: int = 10) -> str:
    """Search for content within documents using semantic similarity.
    Use this for ANY content question - it searches across ALL documents automatically.
    
    Args:
        query: The search query (e.g., "address", "legal terms", "health information")
        limit: Maximum results, default 10 (OPTIONAL - you can omit this)
    
    IMPORTANT: Only use 'query' and optionally 'limit'. Do NOT add other parameters.
    
    Returns:
        Pre-formatted string with search results including filename and content
    """
    results = db.semantic_search(query, limit)
    if not results:
        return "No relevant results found."
    return format_search_results(results)


@agent.tool
def count_files(ctx: RunContext) -> int:
    """Count the total number of unique files in the database.
    
    Returns:
        Number of unique files
    """
    return db.count_files()


@agent.tool
def find_file_by_name(ctx: RunContext, name: str) -> str:
    """Find files by name (partial match, case-insensitive).
    
    Args:
        name: File name or partial name to search for
    
    Returns:
        Pre-formatted string with matching files (include as-is in response)
    """
    files = db.find_file_by_name(name)
    if not files:
        return "No files found matching that name."
    
    # Convert to FileInfo objects for formatting
    file_infos = [FileInfo(**f) for f in files]
    return format_file_tree(file_infos)


@agent.tool
def list_files_by_date(ctx: RunContext, days_ago: int = 1) -> str:
    """List files created within the last N days.
    
    Args:
        days_ago: Number of days to look back (default 1)
    
    Returns:
        Pre-formatted string with files (include as-is in response)
    """
    files = db.list_files_by_date(days_ago)
    if not files:
        return f"No files found from the last {days_ago} day(s)."
    
    file_infos = [FileInfo(**f) for f in files]
    return format_file_tree(file_infos)


@agent.tool
def get_files_by_tag(ctx: RunContext, tag: str) -> str:
    """Get files with a specific tag/folder.
    
    Args:
        tag: Tag name (folder name) to filter by
    
    Returns:
        Pre-formatted string with files (include as-is in response)
    """
    files = db.get_files_by_tag(tag)
    if not files:
        return f"No files found with tag '{tag}'."
    
    file_infos = [FileInfo(**f) for f in files]
    return format_file_tree(file_infos)


@agent.tool
def get_files_by_type(ctx: RunContext, file_type: str) -> str:
    """Get files of a specific type.
    
    Args:
        file_type: File type (pdf, txt, pdf_ocr, image_ocr, html, docx, etc)
    
    Returns:
        Pre-formatted string with files (include as-is in response)
    """
    files = db.get_files_by_type(file_type)
    if not files:
        return f"No files found of type '{file_type}'."
    
    file_infos = [FileInfo(**f) for f in files]
    return format_file_tree(file_infos)


@agent.tool
def get_database_stats(ctx: RunContext) -> str:
    """Get comprehensive database statistics (counts only, not file names).
    
    Returns:
        Pre-formatted string with statistics (include as-is in response)
    """
    stats = db.get_database_stats()
    if 'error' in stats:
        return stats['error']
    return format_stats(stats)


@agent.tool
def list_all_files(ctx: RunContext) -> str:
    """List ALL files in the database with complete metadata.
    Use this when user asks to "list files", "show all files", "what files do I have", etc.
    
    Returns:
        Pre-formatted string with all files (include as-is in response)
    """
    files = db.list_all_files()
    if not files:
        return "No files found in database."
    
    file_infos = [FileInfo(**f) for f in files]
    return format_file_tree(file_infos)


@agent.tool
def get_file_content(ctx: RunContext, file_name: str) -> str:
    """Get the complete text content of a specific file.
    Use this to read file contents, summarize documents, or answer questions about specific files.
    
    Args:
        file_name: Exact name of the file (use list_all_files first to get exact names)
    
    Returns:
        Complete text content of the file with chunk information
    """
    chunks = db.get_file_content(file_name)
    if not chunks:
        return f"File '{file_name}' not found. Use list_all_files to see available files."
    
    output = []
    output.append(f"📄 Content of '{file_name}':")
    output.append(f"Total chunks: {chunks[0]['total_chunks']}\n")
    
    for chunk in chunks:
        output.append(f"--- Chunk {chunk['chunk_index'] + 1}/{chunk['total_chunks']} ---")
        output.append(chunk['text'])
        output.append("")
    
    return "\n".join(output)


@agent.tool
def summarize_all_files(ctx: RunContext) -> str:
    """Get content of ALL files in the database for summarization.
    Use this when user asks to "summarize each file", "give me a summary of all files", etc.
    Returns file names with content (up to 1000 chars per file) for you to summarize.
    
    Returns:
        Structured data with file names and content. You MUST read the content and provide actual summaries.
    """
    all_content = db.get_all_files_content()
    if not all_content:
        return "No files found in database."
    
    output = []
    output.append("📚 Files to Summarize:")
    output.append("")
    
    files = list(all_content.items())
    for i, (file_name, chunks) in enumerate(files, 1):
        is_last = i == len(files)
        prefix = "└── " if is_last else "├── "
        
        output.append(f"{prefix}FILE: {file_name}")
        
        indent = "    " if is_last else "│   "
        
        # Get content (up to 1000 chars for summarization)
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


def format_file_tree(files: List[FileInfo]) -> str:
    """Format files as a tree structure."""
    if not files:
        return "No files found."
    
    output = []
    output.append("📁 Files:")
    for i, file in enumerate(files, 1):
        is_last = i == len(files)
        prefix = "└── " if is_last else "├── "
        
        output.append(f"{prefix}{file.file_name}")
        
        # Add metadata with proper indentation
        indent = "    " if is_last else "│   "
        output.append(f"{indent}├─ Type: {file.file_type}")
        output.append(f"{indent}├─ Path: {file.source_path}")
        output.append(f"{indent}├─ Tags: {', '.join(file.tags)}")
        if file.total_chunks:
            output.append(f"{indent}├─ Chunks: {file.total_chunks}")
        output.append(f"{indent}└─ Modified: {file.timestamp}")
        
        if not is_last:
            output.append("│")
    
    return "\n".join(output)


def format_search_results(results: List[Dict[str, Any]]) -> str:
    """Format search results as a tree structure."""
    if not results:
        return "No results found."
    
    output = []
    output.append("🔍 Search Results:")
    for i, result in enumerate(results, 1):
        is_last = i == len(results)
        prefix = "└── " if is_last else "├── "
        
        # Get relevance score
        score = result.get('_distance', 0)
        relevance = f"{(1 - score) * 100:.1f}%" if score < 1 else "N/A"
        
        output.append(f"{prefix}Result {i} (Relevance: {relevance})")
        
        indent = "    " if is_last else "│   "
        output.append(f"{indent}├─ File: {result.get('file_name', 'Unknown')}")
        output.append(f"{indent}├─ Chunk: {result.get('chunk_index', 0) + 1}/{result.get('total_chunks', 1)}")
        output.append(f"{indent}└─ Text:")
        
        # Format text with indentation
        text = result.get('text', '').strip()
        text_lines = text.split('\n')
        text_indent = f"{indent}    "
        for line in text_lines[:5]:  # Limit to 5 lines
            output.append(f"{text_indent}{line}")
        if len(text_lines) > 5:
            output.append(f"{text_indent}...")
        
        if not is_last:
            output.append("│")
    
    return "\n".join(output)


def format_stats(stats: Dict[str, Any]) -> str:
    """Format database statistics as a tree structure."""
    output = []
    output.append("📊 Database Statistics:")
    output.append(f"├── Total Chunks: {stats.get('total_chunks', 0)}")
    output.append(f"├── Unique Files: {stats.get('unique_files', 0)}")
    
    if 'file_types' in stats:
        output.append("├── File Types:")
        types = stats['file_types']
        for i, (ftype, count) in enumerate(types.items(), 1):
            is_last_type = i == len(types)
            type_prefix = "│   └── " if is_last_type else "│   ├── "
            output.append(f"{type_prefix}{ftype}: {count}")
    
    if 'tags' in stats:
        output.append("└── Tags:")
        tags = stats['tags']
        for i, (tag, count) in enumerate(tags.items(), 1):
            is_last_tag = i == len(tags)
            tag_prefix = "    └── " if is_last_tag else "    ├── "
            output.append(f"{tag_prefix}{tag}: {count}")
    
    return "\n".join(output)


def main():
    """Interactive CLI for querying documents."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Query your document knowledge base")
    parser.add_argument("query", nargs="?", help="Your question")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()
    
    if db.table is None:
        print("No database found. Please run 'make ingest' first to index your documents.")
        return
    
    print("=" * 60)
    print("RAG Query Assistant (LM Studio + PydanticAI)")
    print("=" * 60)
    print(f"Model: qwen3-14b")
    print(f"Database: {db.count_files()} files indexed")
    print()
    
    if args.query:
        # Single query mode
        print(f"Q: {args.query}\n")
        result = agent.run_sync(args.query)
        
        # Simple text output
        print(f"A: {result.output}\n")
    
    elif args.interactive:
        # Interactive mode
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
                
                # Simple text output
                print(f"A: {result.output}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")
    
    else:
        # Show usage
        parser.print_help()
        print("\nExamples:")
        print("  query.py 'How many files do I have?'")
        print("  query.py 'What does document X say about topic Y?'")
        print("  query.py 'Show me PDFs from last week'")
        print("  query.py --interactive")


if __name__ == "__main__":
    main()
