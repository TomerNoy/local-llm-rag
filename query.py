#!/usr/bin/env python3
"""
Query Pipeline for RAG
Uses PydanticAI with hybrid approach: semantic search + metadata queries.
"""

from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta
import os
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
import lancedb
from sentence_transformers import SentenceTransformer
import pyarrow.compute as pc

# Configuration
LANCEDB_PATH = Path("lancedb")
TABLE_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5:7b"  # Good for tool use and instruction following

# Configure Ollama endpoint for OpenAI-compatible API
os.environ.setdefault('OPENAI_BASE_URL', 'http://localhost:11434/v1')
os.environ.setdefault('OPENAI_API_KEY', 'ollama')  # Ollama doesn't need real key


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


# Initialize database
db = DocumentDatabase()


# Define agent with Ollama using OpenAI-compatible endpoint
# Ollama exposes an OpenAI-compatible API at http://localhost:11434
agent = Agent(
    'openai:qwen2.5:7b',
    system_prompt="""You are a helpful document assistant with access to a user's knowledge base.

You have tools for both semantic search (finding content within documents) and metadata queries (file counts, dates, locations, types, tags). Choose the most appropriate tool(s) based on what the user is asking.

When answering:
- CRITICAL: Copy file names and paths EXACTLY as they appear in tool results. Never translate, transliterate, or modify them.
- For non-Latin text (Hebrew, Arabic, etc.), preserve the exact Unicode characters - do not attempt to romanize.
- Format file names in code blocks to preserve encoding: `filename`
- Always cite sources when referencing specific content
- Be concise but complete
- If you need to use multiple tools to answer fully, do so
- Explain if information isn't available or if you need clarification

Your goal is to help users understand and navigate their documents efficiently.""",
)


# Register tools
@agent.tool
def semantic_search(ctx: RunContext, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search for content within documents using semantic similarity.
    
    Args:
        query: The search query
        limit: Maximum number of results (default 5)
    
    Returns:
        List of relevant document chunks with text, file_name, chunk_index, tags, etc.
    """
    return db.semantic_search(query, limit)


@agent.tool
def count_files(ctx: RunContext) -> int:
    """Count the total number of unique files in the database.
    
    Returns:
        Number of unique files
    """
    return db.count_files()


@agent.tool
def find_file_by_name(ctx: RunContext, name: str) -> List[Dict[str, Any]]:
    """Find files by name (partial match, case-insensitive).
    
    Args:
        name: File name or partial name to search for
    
    Returns:
        List of matching files with metadata
    """
    return db.find_file_by_name(name)


@agent.tool
def list_files_by_date(ctx: RunContext, days_ago: int = 1) -> List[Dict[str, Any]]:
    """List files created within the last N days.
    
    Args:
        days_ago: Number of days to look back (default 1)
    
    Returns:
        List of files created within timeframe
    """
    return db.list_files_by_date(days_ago)


@agent.tool
def get_files_by_tag(ctx: RunContext, tag: str) -> List[Dict[str, Any]]:
    """Get files with a specific tag/folder.
    
    Args:
        tag: Tag name (folder name) to filter by
    
    Returns:
        List of files with that tag
    """
    return db.get_files_by_tag(tag)


@agent.tool
def get_files_by_type(ctx: RunContext, file_type: str) -> List[Dict[str, Any]]:
    """Get files of a specific type.
    
    Args:
        file_type: File type (pdf, txt, pdf_ocr, image_ocr, html, docx, etc)
    
    Returns:
        List of files of that type
    """
    return db.get_files_by_type(file_type)


@agent.tool
def get_database_stats(ctx: RunContext) -> Dict[str, Any]:
    """Get comprehensive database statistics (counts only, not file names).
    
    Returns:
        Dictionary with total_chunks, unique_files, file_types breakdown, tags
    """
    return db.get_database_stats()


@agent.tool
def list_all_files(ctx: RunContext) -> List[Dict[str, Any]]:
    """List ALL files in the database with complete metadata.
    Use this when user asks to "list files", "show all files", "what files do I have", etc.
    
    Returns:
        List of all files with file_name, file_type, source_path, tags, timestamp, total_chunks
    """
    return db.list_all_files()


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
    print("RAG Query Assistant (Ollama + PydanticAI)")
    print("=" * 60)
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Database: {db.count_files()} files indexed")
    print()
    
    if args.query:
        # Single query mode
        print(f"Q: {args.query}\n")
        result = agent.run_sync(args.query)
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
