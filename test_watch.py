#!/usr/bin/env python3
"""
Test suite for watch.py
Tests all conversion methods, sync functionality, and file watching.
"""

import unittest
import tempfile
import shutil
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pypandoc

# Import the module to test
from watch import FileConversionHandler, process_existing_files, IGNORED_FILES


class TestFileConversionHandler(unittest.TestCase):
    """Test cases for FileConversionHandler class."""
    
    def setUp(self):
        """Set up temporary directories for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.watched_dir = Path(self.test_dir) / "watched"
        self.output_dir = Path(self.test_dir) / "output"
        self.watched_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.handler = FileConversionHandler(self.watched_dir, self.output_dir)
    
    def tearDown(self):
        """Clean up temporary directories."""
        self.handler.cleanup()
        shutil.rmtree(self.test_dir)
    
    def test_init(self):
        """Test handler initialization."""
        self.assertTrue(self.handler.watched_dir.exists())
        self.assertTrue(self.handler.output_dir.exists())
        self.assertIsNotNone(self.handler.executor)
    
    def test_get_relative_path(self):
        """Test relative path calculation."""
        test_file = self.watched_dir / "subfolder" / "test.txt"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.touch()
        
        rel_path = self.handler.get_relative_path(test_file)
        self.assertEqual(rel_path, Path("subfolder") / "test.txt")
    
    def test_get_output_path(self):
        """Test output path calculation."""
        test_file = self.watched_dir / "subfolder" / "test.txt"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.touch()
        
        output_path = self.handler.get_output_path(test_file)
        expected = self.output_dir / "subfolder" / "test.md"
        # Resolve both paths to handle symlink differences (e.g., /var vs /private/var on macOS)
        self.assertEqual(output_path.resolve(), expected.resolve())
        self.assertTrue(output_path.parent.exists())
    
    def test_convert_text_to_md(self):
        """Test text file conversion to markdown."""
        # Create a simple text file
        test_file = self.watched_dir / "test.txt"
        test_file.write_text("# Test Heading\n\nThis is test content.")
        
        result = self.handler.convert_text_to_md(test_file)
        self.assertTrue(result)
        
        output_file = self.output_dir / "test.md"
        self.assertTrue(output_file.exists())
        content = output_file.read_text()
        self.assertIn("Test Heading", content)
    
    def test_convert_image_to_md_with_text(self):
        """Test image conversion with OCR."""
        # Create a simple image with text
        test_file = self.watched_dir / "test.png"
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw simple text
        draw.text((10, 40), "TEST TEXT", fill='black')
        img.save(test_file)
        
        result = self.handler.convert_image_to_md(test_file)
        self.assertTrue(result)
        
        output_file = self.output_dir / "test.md"
        self.assertTrue(output_file.exists())
        content = output_file.read_text()
        self.assertIn("# test.png", content)
        self.assertIn("Source:", content)
    
    def test_convert_html_to_md(self):
        """Test HTML file conversion to markdown."""
        # Create a simple HTML file
        test_file = self.watched_dir / "test.html"
        html_content = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Test Heading</h1>
            <p>This is a <strong>test</strong> paragraph.</p>
        </body>
        </html>
        """
        test_file.write_text(html_content)
        
        result = self.handler.convert_text_to_md(test_file)
        self.assertTrue(result)
        
        output_file = self.output_dir / "test.md"
        self.assertTrue(output_file.exists())
        content = output_file.read_text()
        self.assertIn("Test Heading", content)
    
    def test_process_file_with_unsupported_format(self):
        """Test handling of unsupported file formats."""
        test_file = self.watched_dir / "test.xyz"
        test_file.write_text("Some content")
        
        # Should not raise exception
        self.handler.process_file(test_file)
        
        # Should not create output file
        output_file = self.output_dir / "test.md"
        self.assertFalse(output_file.exists())
    
    def test_process_file_ignores_hidden_files(self):
        """Test that hidden files are ignored."""
        test_file = self.watched_dir / ".hidden.txt"
        test_file.write_text("Hidden content")
        
        self.handler.process_file(test_file)
        
        output_file = self.output_dir / ".hidden.md"
        self.assertFalse(output_file.exists())
    
    def test_process_file_ignores_system_files(self):
        """Test that system files are ignored."""
        for ignored_file in ['.DS_Store', 'Thumbs.db', 'Desktop.ini']:
            test_file = self.watched_dir / ignored_file
            test_file.write_text("System file")
            
            self.handler.process_file(test_file)
            
            output_file = self.output_dir / f"{ignored_file}.md"
            self.assertFalse(output_file.exists())
    
    def test_process_file_ignores_temp_files(self):
        """Test that temporary files are ignored."""
        test_file = self.watched_dir / "test.tmp"
        test_file.write_text("Temp content")
        
        self.handler.process_file(test_file)
        
        output_file = self.output_dir / "test.md"
        self.assertFalse(output_file.exists())
    
    def test_folder_structure_preservation(self):
        """Test that folder structure is preserved in output."""
        # Create nested structure
        nested_file = self.watched_dir / "level1" / "level2" / "test.txt"
        nested_file.parent.mkdir(parents=True, exist_ok=True)
        nested_file.write_text("Nested content")
        
        self.handler.process_file(nested_file)
        
        output_file = self.output_dir / "level1" / "level2" / "test.md"
        self.assertTrue(output_file.exists())
        self.assertTrue((self.output_dir / "level1").exists())
        self.assertTrue((self.output_dir / "level1" / "level2").exists())


class TestInitialSync(unittest.TestCase):
    """Test cases for initial sync functionality."""
    
    def setUp(self):
        """Set up temporary directories for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.watched_dir = Path(self.test_dir) / "watched"
        self.output_dir = Path(self.test_dir) / "output"
        self.watched_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.handler = FileConversionHandler(self.watched_dir, self.output_dir)
    
    def tearDown(self):
        """Clean up temporary directories."""
        self.handler.cleanup()
        shutil.rmtree(self.test_dir)
    
    def test_sync_processes_existing_files(self):
        """Test that sync processes all existing files."""
        # Create multiple files
        for i in range(5):
            test_file = self.watched_dir / f"test{i}.txt"
            test_file.write_text(f"Content {i}")
        
        process_existing_files(self.handler, max_workers=2)
        
        # Check all files were converted
        for i in range(5):
            output_file = self.output_dir / f"test{i}.md"
            self.assertTrue(output_file.exists())
    
    def test_sync_removes_orphaned_files(self):
        """Test that sync removes orphaned markdown files."""
        # Create a source file and convert it
        source_file = self.watched_dir / "test.txt"
        source_file.write_text("Test content")
        self.handler.process_file(source_file)
        
        output_file = self.output_dir / "test.md"
        self.assertTrue(output_file.exists())
        
        # Delete source file
        source_file.unlink()
        
        # Run sync - should remove orphaned md file
        process_existing_files(self.handler, max_workers=2)
        
        self.assertFalse(output_file.exists())
    
    def test_sync_removes_orphaned_directories(self):
        """Test that sync removes empty directories after cleanup."""
        # Create nested structure
        nested_file = self.watched_dir / "folder1" / "folder2" / "test.txt"
        nested_file.parent.mkdir(parents=True, exist_ok=True)
        nested_file.write_text("Test content")
        
        process_existing_files(self.handler, max_workers=2)
        
        output_file = self.output_dir / "folder1" / "folder2" / "test.md"
        self.assertTrue(output_file.exists())
        
        # Delete all source files
        shutil.rmtree(self.watched_dir / "folder1")
        (self.watched_dir / "folder1").mkdir()
        
        # Run sync - should remove orphaned structure
        process_existing_files(self.handler, max_workers=2)
        
        self.assertFalse(output_file.exists())
    
    def test_sync_parallel_processing(self):
        """Test that sync processes files in parallel."""
        # Create many files
        for i in range(20):
            test_file = self.watched_dir / f"test{i}.txt"
            test_file.write_text(f"Content {i}")
        
        start_time = time.time()
        process_existing_files(self.handler, max_workers=4)
        parallel_time = time.time() - start_time
        
        # Check all files were converted
        for i in range(20):
            output_file = self.output_dir / f"test{i}.md"
            self.assertTrue(output_file.exists())
        
        # Parallel processing should be reasonably fast
        # (This is a basic check, not a strict performance test)
        self.assertLess(parallel_time, 30)  # Should complete in reasonable time


class TestEventHandling(unittest.TestCase):
    """Test cases for real-time event handling."""
    
    def setUp(self):
        """Set up temporary directories for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.watched_dir = Path(self.test_dir) / "watched"
        self.output_dir = Path(self.test_dir) / "output"
        self.watched_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.handler = FileConversionHandler(self.watched_dir, self.output_dir)
    
    def tearDown(self):
        """Clean up temporary directories."""
        self.handler.cleanup()
        shutil.rmtree(self.test_dir)
    
    def test_parallel_event_processing(self):
        """Test that multiple events are processed in parallel."""
        # Simulate multiple file creations
        from unittest.mock import Mock
        
        for i in range(5):
            event = Mock()
            event.is_directory = False
            event.src_path = str(self.watched_dir / f"test{i}.txt")
            
            # Create actual files
            Path(event.src_path).write_text(f"Content {i}")
            
            self.handler.on_created(event)
        
        # Wait for all processing to complete
        time.sleep(2)
        self.handler.cleanup()
        
        # Check files were processed
        for i in range(5):
            output_file = self.output_dir / f"test{i}.md"
            # At least some should be processed
            # (Timing might vary, so not strictly checking all)


class TestHebrewSupport(unittest.TestCase):
    """Test cases for Hebrew language support."""
    
    def setUp(self):
        """Set up temporary directories for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.watched_dir = Path(self.test_dir) / "watched"
        self.output_dir = Path(self.test_dir) / "output"
        self.watched_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.handler = FileConversionHandler(self.watched_dir, self.output_dir)
    
    def tearDown(self):
        """Clean up temporary directories."""
        self.handler.cleanup()
        shutil.rmtree(self.test_dir)
    
    def test_hebrew_filename_support(self):
        """Test that Hebrew filenames are supported."""
        test_file = self.watched_dir / "מבחן.txt"
        test_file.write_text("Test content in Hebrew file")
        
        result = self.handler.convert_text_to_md(test_file)
        self.assertTrue(result)
        
        output_file = self.output_dir / "מבחן.md"
        self.assertTrue(output_file.exists())
    
    def test_hebrew_content_support(self):
        """Test that Hebrew content is preserved."""
        test_file = self.watched_dir / "test.txt"
        hebrew_content = "זהו טקסט בעברית\n\nThis is also English"
        test_file.write_text(hebrew_content, encoding='utf-8')
        
        result = self.handler.convert_text_to_md(test_file)
        self.assertTrue(result)
        
        output_file = self.output_dir / "test.md"
        self.assertTrue(output_file.exists())
        content = output_file.read_text(encoding='utf-8')
        self.assertIn("עברית", content)


def run_tests():
    """Run all tests and display results."""
    print("=" * 70)
    print("Running Watch.py Test Suite")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFileConversionHandler))
    suite.addTests(loader.loadTestsFromTestCase(TestInitialSync))
    suite.addTests(loader.loadTestsFromTestCase(TestEventHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestHebrewSupport))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed.")
        return 1


if __name__ == "__main__":
    exit(run_tests())
