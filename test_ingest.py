#!/usr/bin/env python3
"""
Test suite for ingest.py
Tests ingestion, hash-based change detection, orphan cleanup, validation, and deletion.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from ingest import DocumentIngester, MD_CONTENT_DIR, WATCHED_DIR


class TestDocumentIngestion(unittest.TestCase):
    """Test document ingestion functionality."""
    
    def setUp(self):
        """Set up test environment with temporary directories."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_md_dir = self.test_dir / "md-content"
        self.test_watched_dir = self.test_dir / "watched-dir"
        self.test_db_path = self.test_dir / "test-lancedb"
        
        self.test_md_dir.mkdir(parents=True)
        self.test_watched_dir.mkdir(parents=True)
        
        # Monkey patch paths
        self.original_md_dir = MD_CONTENT_DIR
        self.original_watched_dir = WATCHED_DIR
        
        import ingest
        ingest.MD_CONTENT_DIR = self.test_md_dir
        ingest.WATCHED_DIR = self.test_watched_dir
        
        # Create ingester
        self.ingester = DocumentIngester(
            db_path=self.test_db_path,
            table_name="test_documents"
        )
    
    def tearDown(self):
        """Clean up test environment."""
        import ingest
        ingest.MD_CONTENT_DIR = self.original_md_dir
        ingest.WATCHED_DIR = self.original_watched_dir
        
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def create_test_file(self, rel_path: str, content: str):
        """Helper to create a test markdown file."""
        md_path = self.test_md_dir / rel_path
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(content, encoding='utf-8')
        
        # Create corresponding source file
        source_path = self.test_watched_dir / Path(rel_path).with_suffix('.txt')
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(content, encoding='utf-8')
        
        return md_path, source_path
    
    def test_basic_ingestion(self):
        """Test basic file ingestion."""
        content = "This is a test document.\n\nIt has multiple paragraphs."
        md_path, source_path = self.create_test_file("test1.md", content)
        
        # Ingest
        chunks = self.ingester.ingest_file(md_path)
        
        # Verify
        self.assertGreater(chunks, 0, "Should create at least one chunk")
        stats = self.ingester.get_stats()
        self.assertEqual(stats['total_chunks'], chunks)
        self.assertEqual(stats['unique_files'], 1)
    
    def test_hash_based_skip(self):
        """Test that unchanged files are skipped."""
        content = "Test content for hash checking."
        md_path, source_path = self.create_test_file("test2.md", content)
        
        # First ingestion
        chunks1 = self.ingester.ingest_file(md_path)
        self.assertGreater(chunks1, 0)
        
        # Second ingestion without changes - should skip
        chunks2 = self.ingester.ingest_file(md_path)
        self.assertEqual(chunks2, -1, "Should return -1 for skipped file")
        
        # Verify no duplicate chunks
        stats = self.ingester.get_stats()
        self.assertEqual(stats['total_chunks'], chunks1)
    
    def test_update_modified_file(self):
        """Test updating a file with different hash."""
        md_path, source_path = self.create_test_file("test3.md", "Original content")
        
        # First ingestion
        chunks1 = self.ingester.ingest_file(md_path)
        self.assertGreater(chunks1, 0)
        
        # Modify file
        md_path.write_text("Modified content - completely different!", encoding='utf-8')
        
        # Re-ingest - should delete old and create new
        chunks2 = self.ingester.ingest_file(md_path)
        self.assertGreater(chunks2, 0)
        
        # Verify only new chunks exist (old ones deleted)
        stats = self.ingester.get_stats()
        self.assertEqual(stats['total_chunks'], chunks2)
    
    def test_orphan_cleanup(self):
        """Test cleanup of orphaned database entries."""
        # Create and ingest two files
        md1, src1 = self.create_test_file("file1.md", "Content 1")
        md2, src2 = self.create_test_file("file2.md", "Content 2")
        
        self.ingester.ingest_file(md1)
        self.ingester.ingest_file(md2)
        
        stats_before = self.ingester.get_stats()
        self.assertEqual(stats_before['unique_files'], 2)
        
        # Delete one markdown file (simulating manual deletion)
        md2.unlink()
        src2.unlink()
        
        # Run cleanup
        md_files = list(self.test_md_dir.rglob("*.md"))
        deleted = self.ingester.cleanup_orphaned_chunks(md_files)
        
        # Verify orphaned chunks were removed
        self.assertGreater(deleted, 0, "Should delete orphaned chunks")
        stats_after = self.ingester.get_stats()
        self.assertEqual(stats_after['unique_files'], 1)
    
    def test_delete_file_api(self):
        """Test the delete_file API."""
        md_path, source_path = self.create_test_file("test4.md", "Content to delete")
        
        # Ingest
        chunks = self.ingester.ingest_file(md_path)
        self.assertGreater(chunks, 0)
        
        # Debug: check what's in the database
        table = self.ingester.table.to_arrow()
        stored_paths = table['source_path'].to_pylist()
        print(f"\nStored paths: {stored_paths}")
        print(f"Looking for: {source_path}")
        print(f"Match: {str(source_path) in stored_paths}")
        
        # Delete via API
        deleted = self.ingester.delete_file(source_path)
        self.assertEqual(deleted, chunks, "Should delete all chunks for the file")
        
        # Verify deletion
        stats = self.ingester.get_stats()
        self.assertEqual(stats['total_chunks'], 0)
    
    def test_ingest_all_with_sync(self):
        """Test ingest_all performs bidirectional sync."""
        # Create files
        self.create_test_file("a.md", "File A")
        self.create_test_file("b.md", "File B")
        self.create_test_file("subdir/c.md", "File C")
        
        # First run
        stats1 = self.ingester.ingest_all()
        self.assertEqual(stats1['processed'], 3)
        
        # Delete one file and add another
        (self.test_md_dir / "b.md").unlink()
        (self.test_watched_dir / "b.txt").unlink()
        self.create_test_file("d.md", "File D")
        
        # Second run - should clean orphans and process new file
        stats2 = self.ingester.ingest_all()
        
        # Verify: 2 skipped (a, c unchanged), 1 processed (d), orphans deleted
        self.assertEqual(stats2['skipped'], 2, "Should skip unchanged files")
        self.assertEqual(stats2['processed'], 1, "Should process new file")
        self.assertGreater(stats2['deleted'], 0, "Should delete orphaned chunks")
        
        # Final check
        final_stats = self.ingester.get_stats()
        self.assertEqual(final_stats['unique_files'], 3)  # a, c, d
    
    def test_cleanup_invalid_chunks_missing_hash(self):
        """Test cleanup of chunks with missing hash field."""
        md_path, source_path = self.create_test_file("test_invalid.md", "Test content")
        
        # Ingest normally
        chunks = self.ingester.ingest_file(md_path)
        self.assertGreater(chunks, 0)
        
        # Manually corrupt data by setting hash to empty
        table = self.ingester.table.to_arrow()
        import pyarrow as pa
        
        # Create corrupted record with empty hash
        corrupted_table = table.set_column(
            table.schema.get_field_index('file_hash'),
            'file_hash',
            pa.array([''] * len(table))
        )
        
        # Replace table
        self.ingester.db.drop_table(self.ingester.table_name)
        self.ingester.table = self.ingester.db.create_table(
            self.ingester.table_name,
            corrupted_table
        )
        
        # Run cleanup
        deleted = self.ingester.cleanup_invalid_chunks()
        self.assertGreater(deleted, 0, "Should delete chunks with empty hash")
        
        # Verify all chunks removed
        stats = self.ingester.get_stats()
        self.assertEqual(stats['total_chunks'], 0)
    
    def test_cleanup_invalid_chunks_empty_hash(self):
        """Test cleanup of chunks with empty hash values via ingest_all."""
        md_path, source_path = self.create_test_file("test_empty.md", "Test content for empty hash")
        
        # Ingest normally
        chunks = self.ingester.ingest_file(md_path)
        self.assertGreater(chunks, 0)
        
        # Manually corrupt data by setting hash to empty string
        table = self.ingester.table.to_arrow()
        import pyarrow as pa
        
        # Create corrupted record with empty hash (LanceDB doesn't support nulls in string fields)
        corrupted_table = table.set_column(
            table.schema.get_field_index('file_hash'),
            'file_hash',
            pa.array([''] * len(table))
        )
        
        # Replace table (drop and recreate)
        self.ingester.db.drop_table(self.ingester.table_name)
        self.ingester.table = self.ingester.db.create_table(
            self.ingester.table_name,
            corrupted_table
        )
        
        # Reopen the table to ensure fresh connection
        self.ingester.table = self.ingester.db.open_table(self.ingester.table_name)
        
        # Run ingest_all which includes cleanup_invalid_chunks
        # This should detect and remove the corrupted chunks, then reingest
        stats = self.ingester.ingest_all()
        
        # Verify: should have deleted invalid chunks and reingested the file
        self.assertGreater(stats['deleted'], 0, "Should delete chunks with empty hash")
        self.assertEqual(stats['processed'], 1, "Should reingest the file")
        
        # Verify all chunks now have valid hashes
        final_table = self.ingester.table.to_arrow()
        hashes = final_table['file_hash'].to_pylist()
        self.assertTrue(all(h and h != '' for h in hashes), "All chunks should have valid hashes")
    
    def test_ingest_all_runs_validation(self):
        """Test that ingest_all runs validation before processing."""
        # Create file and ingest
        md_path, source_path = self.create_test_file("test_validation.md", "Content")
        self.ingester.ingest_file(md_path)
        
        # Corrupt the hash
        table = self.ingester.table.to_arrow()
        import pyarrow as pa
        
        corrupted_table = table.set_column(
            table.schema.get_field_index('file_hash'),
            'file_hash',
            pa.array([''] * len(table))
        )
        
        self.ingester.db.drop_table(self.ingester.table_name)
        self.ingester.table = self.ingester.db.create_table(
            self.ingester.table_name,
            corrupted_table
        )
        
        # Run ingest_all - should clean invalid chunks and reingest
        stats = self.ingester.ingest_all()
        
        # Verify: old invalid chunks deleted, new valid chunks created
        self.assertGreater(stats['deleted'], 0, "Should delete invalid chunks")
        self.assertEqual(stats['processed'], 1, "Should reingest the file")
        
        # Verify all chunks now have valid hashes
        final_table = self.ingester.table.to_arrow()
        hashes = final_table['file_hash'].to_pylist()
        self.assertTrue(all(h and h != '' for h in hashes), "All chunks should have valid hashes")
    
    def test_force_reingest(self):
        """Test force flag reingests all files."""
        self.create_test_file("test5.md", "Test content")
        
        # First ingestion
        stats1 = self.ingester.ingest_all(force=False)
        self.assertEqual(stats1['processed'], 1)
        
        # Second ingestion without force - should skip
        stats2 = self.ingester.ingest_all(force=False)
        self.assertEqual(stats2['skipped'], 1)
        self.assertEqual(stats2['processed'], 0)
        
        # Third ingestion with force - should reprocess
        stats3 = self.ingester.ingest_all(force=True)
        self.assertEqual(stats3['processed'], 1)
        self.assertEqual(stats3['skipped'], 0)


class TestHashCalculation(unittest.TestCase):
    """Test file hashing functionality."""
    
    def test_consistent_hash(self):
        """Test that same content produces same hash."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Test content")
            path = Path(f.name)
        
        try:
            ingester = DocumentIngester(
                db_path=Path(tempfile.mkdtemp()) / "test-db"
            )
            hash1 = ingester.get_file_hash(path)
            hash2 = ingester.get_file_hash(path)
            
            self.assertEqual(hash1, hash2, "Same file should produce same hash")
        finally:
            path.unlink()
    
    def test_different_hash_for_modified(self):
        """Test that modified content produces different hash."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Original")
            path = Path(f.name)
        
        try:
            ingester = DocumentIngester(
                db_path=Path(tempfile.mkdtemp()) / "test-db"
            )
            hash1 = ingester.get_file_hash(path)
            
            # Modify file
            path.write_text("Modified")
            hash2 = ingester.get_file_hash(path)
            
            self.assertNotEqual(hash1, hash2, "Modified file should have different hash")
        finally:
            path.unlink()


def main():
    """Run all tests."""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
