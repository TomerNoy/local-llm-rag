#!/usr/bin/env python3
"""
Ingest Pipeline for RAG
Processes md-content and stores chunks with embeddings in LanceDB.
"""

import os
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import lancedb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
MD_CONTENT_DIR = Path("md-content")
WATCHED_DIR = Path("watched-dir")
LANCEDB_PATH = Path("lancedb")
TABLE_NAME = "documents"

# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, fast and good quality


class DocumentIngester:
    """Handles document ingestion into LanceDB."""
    
    @staticmethod
    def normalize_path(path: Path) -> str:
        """Normalize path to handle macOS symlinks consistently."""
        # resolve() follows symlinks (/var -> /private/var on macOS)
        resolved = path.resolve()
        # Convert to string for consistent comparison
        return str(resolved)
    
    def __init__(
        self,
        db_path: Path = LANCEDB_PATH,
        table_name: str = TABLE_NAME,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        embedding_model: str = EMBEDDING_MODEL
    ):
        self.db_path = db_path
        self.table_name = table_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}...")
        self.embedder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        print(f"✓ Model loaded (dimension: {self.embedding_dim})")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Connect to LanceDB
        self.db = lancedb.connect(str(db_path))
        self._init_table()
    
    def _init_table(self):
        """Initialize or get existing table."""
        try:
            self.table = self.db.open_table(self.table_name)
            print(f"✓ Opened existing table: {self.table_name}")
        except:
            print(f"Creating new table: {self.table_name}")
            # Create with empty schema - will be inferred from first insert
            self.table = None
    
    def get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file for change detection."""
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        return md5.hexdigest()
    
    def extract_tags_from_path(self, md_path: Path) -> List[str]:
        """Extract folder structure as tags."""
        # Get relative path from md-content
        rel_path = md_path.relative_to(MD_CONTENT_DIR)
        # Get all parent folders
        tags = [part for part in rel_path.parent.parts if part != '.']
        return tags if tags else ["root"]
    
    def determine_file_type(self, md_path: Path, source_path: Path) -> str:
        """Determine the type of original file."""
        if not source_path.exists():
            return "unknown"
        
        suffix = source_path.suffix.lower()
        
        # Check if it's from OCR
        with open(md_path, 'r', encoding='utf-8') as f:
            first_lines = f.read(200)
            if "*Source:" in first_lines:
                if suffix == '.pdf':
                    return "pdf_ocr"
                elif suffix in {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}:
                    return "image_ocr"
        
        # Text-based conversions
        if suffix == '.pdf':
            return "pdf_text"
        elif suffix == '.txt':
            return "txt"
        elif suffix in {'.html', '.htm'}:
            return "html"
        elif suffix in {'.doc', '.docx'}:
            return "docx"
        elif suffix in {'.rtf'}:
            return "rtf"
        else:
            return "other"
    
    def find_source_file(self, md_path: Path) -> Path:
        """Find original source file in watched-dir."""
        # Get relative path and reconstruct with different extensions
        rel_path = md_path.relative_to(MD_CONTENT_DIR)
        base_path = WATCHED_DIR / rel_path.with_suffix('')
        
        # Try common extensions
        extensions = ['.pdf', '.txt', '.jpg', '.jpeg', '.png', '.html', '.htm', 
                     '.doc', '.docx', '.rtf', '.tiff', '.bmp', '.odt']
        
        for ext in extensions:
            source_path = base_path.with_suffix(ext)
            if source_path.exists():
                return source_path
        
        # Fallback: return the path (not resolved) even if it doesn't exist
        return base_path
    
    def get_existing_hash(self, source_path: Path) -> str:
        """Get hash of existing file in database, if any."""
        if self.table is None:
            return None
        
        try:
            # Query for existing chunks with this source_path
            source_str = self.normalize_path(source_path)
            table = self.table.to_arrow()
            
            # Filter for matching source_path
            import pyarrow.compute as pc
            mask = pc.equal(table['source_path'], source_str)
            filtered = table.filter(mask)
            
            if len(filtered) > 0:
                return filtered['file_hash'][0].as_py()
        except Exception as e:
            pass
        return None
    
    def delete_file(self, source_path: Path) -> int:
        """Delete all chunks for a given source file."""
        if self.table is None:
            return 0
        
        try:
            source_str = self.normalize_path(source_path)
            
            # Count how many chunks before deletion - refresh table first
            import pyarrow.compute as pc
            table = self.table.to_arrow()
            
            mask = pc.equal(table['source_path'], source_str)
            count_before = int(pc.sum(pc.cast(mask, 'int64')).as_py())
            
            if count_before == 0:
                return 0
            
            # Delete chunks with this source_path (escape quotes properly)
            escaped_path = source_str.replace("'", "''")
            self.table.delete(f"source_path = '{escaped_path}'")
            print(f"✓ Deleted {count_before} chunks for: {source_path.name}")
            return count_before
        except Exception as e:
            print(f"✗ Error deleting {source_path}: {e}")
            return 0
    
    def chunk_document(self, content: str, md_path: Path, source_path: Path, file_hash: str) -> List[Dict[str, Any]]:
        """Split document into chunks and prepare for ingestion."""
        # Split text into chunks
        chunks = self.text_splitter.split_text(content)
        total_chunks = len(chunks)
        
        # Extract metadata
        tags = self.extract_tags_from_path(md_path)
        file_type = self.determine_file_type(md_path, source_path)
        file_name = md_path.stem
        timestamp = datetime.now().isoformat()
        
        # Prepare chunk records
        records = []
        for idx, chunk_text in enumerate(chunks):
            # Generate embedding
            embedding = self.embedder.encode(chunk_text).tolist()
            
            record = {
                "vector": embedding,
                "text": chunk_text,
                "source_path": self.normalize_path(source_path),
                "md_path": self.normalize_path(md_path),
                "file_name": file_name,
                "file_type": file_type,
                "file_hash": file_hash,
                "tags": tags,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "chunk_size": len(chunk_text),
                "timestamp": timestamp
            }
            records.append(record)
        
        return records
    
    def ingest_file(self, md_path: Path, force: bool = False) -> int:
        """Ingest a single markdown file with hash-based change detection."""
        try:
            # Find source file
            source_path = self.find_source_file(md_path)
            
            # Calculate hash of markdown file
            file_hash = self.get_file_hash(md_path)
            
            # Check if file already exists with same hash (unless force=True)
            if not force:
                existing_hash = self.get_existing_hash(source_path)
                if existing_hash == file_hash:
                    print(f"⊘ Skipped (unchanged): {md_path.name}")
                    return -1  # -1 indicates skipped due to unchanged
            
            # Read content
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Skip empty files
            if not content.strip():
                print(f"⊘ Skipped (empty): {md_path.name}")
                return 0
            
            # If file exists but hash is different, delete old chunks first
            if self.table is not None:
                existing_hash = self.get_existing_hash(source_path)
                if existing_hash and existing_hash != file_hash:
                    self.delete_file(source_path)
            
            # Chunk and prepare records
            records = self.chunk_document(content, md_path, source_path, file_hash)
            
            # Add to table
            if self.table is None:
                # Create table with first batch
                self.table = self.db.create_table(self.table_name, records)
            else:
                self.table.add(records)
            
            print(f"✓ Ingested: {md_path.name} ({len(records)} chunks)")
            return len(records)
            
        except Exception as e:
            print(f"✗ Error ingesting {md_path}: {e}")
            return 0
    
    def cleanup_invalid_chunks(self) -> int:
        """Remove chunks that are missing required fields or can't be validated."""
        if self.table is None:
            return 0
        
        try:
            table = self.table.to_arrow()
            import pyarrow.compute as pc
            
            total_deleted = 0
            schema_fields = set(table.schema.names)
            
            # Check 1: Remove chunks missing file_hash field
            if 'file_hash' not in schema_fields:
                # Old schema without hashes - need to delete all and reingest
                count = table.num_rows
                if count > 0:
                    print(f"⚠ Database schema outdated (missing file_hash field)")
                    print(f"  Removing all {count} chunks for clean reingest...")
                    # Delete all rows by creating empty table
                    self.table.delete("chunk_index >= 0")
                    return count
            
            # Check 2: Remove chunks with NULL/empty hash
            try:
                hash_col = table['file_hash']
                
                # Check for both null and empty values
                null_mask = pc.is_null(hash_col)
                
                # For string columns, also check for empty strings
                # Need to handle case where column might have nulls that can't be compared
                try:
                    empty_mask = pc.equal(hash_col, '')
                    invalid_mask = pc.or_(null_mask, empty_mask)
                except:
                    # If equal fails (due to nulls), just use null mask
                    invalid_mask = null_mask
                
                # Count invalid chunks safely
                invalid_sum = pc.sum(pc.cast(invalid_mask, 'int64'))
                invalid_count = int(invalid_sum.as_py() or 0)
                
                if invalid_count > 0:
                    print(f"⚠ Found {invalid_count} chunks with missing hash")
                    # Get md_paths of invalid chunks to delete them
                    invalid_rows = table.filter(invalid_mask)
                    invalid_md_paths = set(invalid_rows['md_path'].to_pylist())
                    
                    for md_path in invalid_md_paths:
                        escaped = md_path.replace("'", "''")
                        try:
                            self.table.delete(f"md_path = '{escaped}'")
                        except Exception as del_err:
                            print(f"⚠ Error deleting {md_path}: {del_err}")
                    
                    print(f"✓ Removed {invalid_count} chunks with invalid hashes")
                    total_deleted += invalid_count
            except Exception as e:
                print(f"⚠ Error checking hashes: {e}")
            
            # Check 3: Remove chunks where md_path doesn't exist
            md_paths_in_db = set(table['md_path'].to_pylist())
            for md_path_str in md_paths_in_db:
                if not Path(md_path_str).exists():
                    # Already handled by cleanup_orphaned_chunks, but double-check
                    pass
            
            return total_deleted
            
        except Exception as e:
            print(f"⚠ Error cleaning invalid chunks: {e}")
            return 0
    
    def cleanup_orphaned_chunks(self, current_md_files: List[Path]) -> int:
        """Remove database entries for MD files that no longer exist."""
        if self.table is None:
            return 0
        
        try:
            # Get all unique md paths in database
            table = self.table.to_arrow()
            db_md_paths = set(table['md_path'].to_pylist())
            
            if not db_md_paths:
                return 0
            
            # Build set of current md paths (normalized)
            current_md_paths = set(self.normalize_path(md_path) for md_path in current_md_files)
            
            # Find orphaned entries (in DB but md file doesn't exist)
            orphaned_md_paths = db_md_paths - current_md_paths
            
            if not orphaned_md_paths:
                return 0
            
            # Delete chunks for orphaned md files
            total_deleted = 0
            import pyarrow.compute as pc
            for md_path_str in orphaned_md_paths:
                # Delete all chunks with this md_path
                escaped_path = md_path_str.replace("'", "''")
                try:
                    # Count before deletion
                    mask = pc.equal(table['md_path'], md_path_str)
                    count = int(pc.sum(pc.cast(mask, 'int64')).as_py())
                    
                    if count > 0:
                        self.table.delete(f"md_path = '{escaped_path}'")
                        print(f"✓ Deleted {count} chunks for removed file: {Path(md_path_str).name}")
                        total_deleted += count
                except Exception as e:
                    print(f"⚠ Error deleting chunks for {md_path_str}: {e}")
            
            return total_deleted
        except Exception as e:
            print(f"⚠ Error cleaning orphaned chunks: {e}")
            return 0
    
    def ingest_all(self, force: bool = False) -> Dict[str, int]:
        """Ingest all markdown files from md-content with bidirectional sync."""
        print("=" * 60)
        print("Document Ingestion Pipeline")
        print("=" * 60)
        print(f"Source: {MD_CONTENT_DIR.absolute()}")
        print(f"Database: {self.db_path.absolute()}")
        print(f"Chunk size: {self.chunk_size} (overlap: {self.chunk_overlap})")
        print("=" * 60)
        
        # Collect all markdown files
        md_files = list(MD_CONTENT_DIR.rglob("*.md"))
        total_files = len(md_files)
        
        print(f"\nFound {total_files} markdown file(s)")
        
        # Step 1: Remove invalid chunks (missing hash, corrupted data)
        print("\n=== Validation: Checking for invalid chunks ===")
        invalid_deleted = self.cleanup_invalid_chunks()
        if invalid_deleted > 0:
            print()
        else:
            print("All chunks valid.\n")
        
        # Step 2: Cleanup orphaned chunks (bidirectional sync)
        # Always run this, even if no files exist (to clean up everything)
        print("=== Sync Check: Cleaning orphaned database entries ===")
        orphaned_deleted = self.cleanup_orphaned_chunks(md_files)
        if orphaned_deleted > 0:
            print(f"✓ Removed {orphaned_deleted} orphaned chunks\n")
        else:
            print("No orphaned chunks found.\n")
        
        total_deleted = invalid_deleted + orphaned_deleted
        
        # Early return if no files to process
        if total_files == 0:
            print("No files to ingest.")
            return {"processed": 0, "chunks": 0, "skipped": 0, "errors": 0, "deleted": total_deleted}
        
        # Step 3: Process files
        stats = {"processed": 0, "chunks": 0, "skipped": 0, "errors": 0, "deleted": total_deleted}
        
        for idx, md_path in enumerate(md_files, 1):
            print(f"[{idx}/{total_files}] ", end="")
            chunks = self.ingest_file(md_path, force=force)
            
            if chunks > 0:
                stats["processed"] += 1
                stats["chunks"] += chunks
            elif chunks == -1:
                stats["skipped"] += 1  # Skipped due to unchanged hash
            elif chunks == 0:
                stats["skipped"] += 1  # Skipped due to empty file
            else:
                stats["errors"] += 1
        
        # Print summary
        print("\n" + "=" * 60)
        print("Ingestion Complete")
        print("=" * 60)
        print(f"✓ Processed: {stats['processed']}/{total_files} files")
        print(f"✓ Total chunks: {stats['chunks']}")
        if stats['skipped'] > 0:
            print(f"⊘ Skipped: {stats['skipped']}")
        if stats['deleted'] > 0:
            print(f"🗑  Deleted orphans: {stats['deleted']}")
        if stats['errors'] > 0:
            print(f"✗ Errors: {stats['errors']}")
        print()
        
        return stats
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        if self.table is None:
            print("No documents in database yet.")
            return []
        
        # Generate query embedding
        query_vector = self.embedder.encode(query).tolist()
        
        # Search
        results = self.table.search(query_vector).limit(limit).to_list()
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if self.table is None:
            return {"total_chunks": 0, "unique_files": 0}
        
        total_chunks = self.table.count_rows()
        # Get unique files by counting distinct md_paths
        table = self.table.to_arrow()
        unique_files = len(set(table['md_path'].to_pylist()))
        
        return {
            "total_chunks": total_chunks,
            "unique_files": unique_files,
            "embedding_dimension": self.embedding_dim
        }
    
    def compact_history(self) -> Dict[str, int]:
        """Compact the database to remove deleted data and clear transaction history."""
        if self.table is None:
            print("No table to compact.")
            return {"before_bytes": 0, "after_bytes": 0, "saved_bytes": 0}
        
        import os
        
        # Calculate size before compaction
        table_path = Path(self.db_path) / f"{self.table_name}.lance"
        before_size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, dn, fn in os.walk(table_path)
            for f in fn
        )
        
        print(f"Compacting table '{self.table_name}'...")
        print(f"Size before: {before_size:,} bytes ({before_size / 1024:.1f} KB)")
        
        try:
            # Try new API first (0.21.0+)
            try:
                self.table.optimize()
                print("✓ Used optimize() method")
            except AttributeError:
                # Fall back to old API
                self.table.compact_files()
                print("✓ Used compact_files() method")
            
            # Cleanup old versions
            try:
                self.table.cleanup_old_versions()
                print("✓ Cleaned up old versions")
            except Exception as e:
                print(f"⚠ Could not cleanup old versions: {e}")
            
            # Calculate size after compaction
            after_size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, dn, fn in os.walk(table_path)
                for f in fn
            )
            
            saved = before_size - after_size
            print(f"Size after: {after_size:,} bytes ({after_size / 1024:.1f} KB)")
            if saved > 0:
                print(f"✓ Saved {saved:,} bytes ({saved / 1024:.1f} KB)")
            else:
                print("✓ Database already optimized")
            
            return {
                "before_bytes": before_size,
                "after_bytes": after_size,
                "saved_bytes": saved
            }
        except Exception as e:
            print(f"⚠ Error during compaction: {e}")
            print("Note: Compaction requires 'pylance' package. Install with: pip install pylance")
            return {"before_bytes": before_size, "after_bytes": before_size, "saved_bytes": 0}


def main():
    """Main ingestion function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest markdown files into LanceDB")
    parser.add_argument("--force", action="store_true", help="Force re-ingestion of all files")
    parser.add_argument("--search", type=str, help="Test search with a query")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--delete", type=str, help="Delete all chunks for a specific source file path")
    parser.add_argument("--compact", action="store_true", help="Compact database to remove deleted data and clear history")
    parser.add_argument("--clear", action="store_true", help="Clear all data and history (deletes and recreates table)")
    args = parser.parse_args()
    
    # Initialize ingester
    ingester = DocumentIngester()
    
    if args.clear:
        # Clear all data and history
        import shutil
        table_path = Path(ingester.db_path) / f"{ingester.table_name}.lance"
        if table_path.exists():
            print(f"Clearing table '{ingester.table_name}'...")
            shutil.rmtree(table_path)
            print(f"✓ Deleted all data and history")
            print(f"✓ Table will be recreated on next ingestion")
        else:
            print("Table already empty.")
        print()
    elif args.compact:
        # Compact the database
        stats = ingester.compact_history()
        print()
    elif args.delete:
        # Delete chunks for a specific file
        source_path = Path(args.delete).resolve()
        print(f"Deleting chunks for: {source_path}")
        deleted = ingester.delete_file(source_path)
        if deleted == 0:
            print("No chunks found for this file.")
    elif args.stats:
        # Show stats
        stats = ingester.get_stats()
        print("=" * 60)
        print("Database Statistics")
        print("=" * 60)
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Unique files: {stats['unique_files']}")
        print(f"Embedding dimension: {stats['embedding_dimension']}")
        print()
    elif args.search:
        # Test search
        print(f"Searching for: {args.search}\n")
        results = ingester.search(args.search, limit=3)
        for idx, result in enumerate(results, 1):
            print(f"{idx}. {result['file_name']} (chunk {result['chunk_index']})")
            print(f"   Type: {result['file_type']}, Tags: {result['tags']}")
            print(f"   {result['text'][:200]}...")
            print()
    else:
        # Ingest documents
        ingester.ingest_all(force=args.force)


if __name__ == "__main__":
    main()
