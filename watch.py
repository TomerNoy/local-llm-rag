#!/usr/bin/env python3
"""
File Watcher for RAG Pipeline
Watches watched-dir and converts files to markdown in md-content/
Preserves folder structure for LanceDB ingestion.
"""

import os
import time
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pypandoc
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# Check if marker-pdf is available (in venv311)
MARKER_AVAILABLE = Path(__file__).parent / 'venv311' / 'bin' / 'marker_single'
if MARKER_AVAILABLE.exists():
    MARKER_AVAILABLE = True
    print("✓ Marker available in venv311 for better Hebrew OCR")
else:
    MARKER_AVAILABLE = False
    print("⚠️  marker-pdf not installed in venv311. Using pytesseract for OCR.")


# Configuration
WATCHED_DIR = Path("watched-dir")
OUTPUT_DIR = Path("md-content")

SUPPORTED_TEXT_FORMATS = {'.txt', '.doc', '.docx', '.rtf', '.html', '.htm', '.odt'}
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
SUPPORTED_PDF = {'.pdf'}

# Files/patterns to ignore
IGNORED_FILES = {'.ds_store', 'thumbs.db', 'desktop.ini', '.localized'}
IGNORED_PATTERNS = {'.tmp', '.temp', '~', '.swp', '.swo', '.bak'}


class FileConversionHandler(FileSystemEventHandler):
    """Handles file system events and converts files to markdown."""
    
    def __init__(self, watched_dir, output_dir, max_workers=3):
        self.watched_dir = Path(watched_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_tasks = {}
    
    def get_relative_path(self, file_path):
        """Get path relative to watched directory."""
        return Path(file_path).resolve().relative_to(self.watched_dir)
    
    def get_output_path(self, file_path):
        """Calculate output path maintaining directory structure."""
        rel_path = self.get_relative_path(file_path)
        output_path = self.output_dir / rel_path.with_suffix('.md')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path
    
    def convert_text_to_md(self, file_path):
        """Convert text-based documents to markdown using pandoc."""
        try:
            output_path = self.get_output_path(file_path)
            
            # Handle plain text files directly (pandoc doesn't support .txt)
            if file_path.suffix.lower() == '.txt':
                content = Path(file_path).read_text(encoding='utf-8')
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✓ Converted: {file_path} -> {output_path}")
                return True
            
            # Use pandoc for other formats
            pypandoc.convert_file(
                str(file_path),
                'md',
                outputfile=str(output_path)
            )
            
            print(f"✓ Converted: {file_path} -> {output_path}")
            return True
        except Exception as e:
            print(f"✗ Error converting {file_path}: {e}")
            return False
    
    def convert_image_to_md(self, file_path):
        """Convert image to markdown using OCR."""
        try:
            output_path = self.get_output_path(file_path)
            
            # Perform OCR with Hebrew and English support
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image, lang='heb+eng')
            
            # Write to markdown with metadata
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# {Path(file_path).name}\n\n")
                f.write(f"*Source: {file_path}*\n\n")
                f.write("---\n\n")
                f.write(text)
            
            print(f"✓ OCR'd: {file_path} -> {output_path}")
            return True
        except Exception as e:
            print(f"✗ Error OCR'ing {file_path}: {e}")
            return False
    
    def convert_pdf_to_md(self, file_path):
        """Convert PDF to markdown (with OCR if needed)."""
        try:
            output_path = self.get_output_path(file_path)
            
            # Try pandoc first (for text-based PDFs)
            try:
                pypandoc.convert_file(
                    str(file_path),
                    'md',
                    outputfile=str(output_path)
                )
                print(f"✓ Converted PDF: {file_path} -> {output_path}")
                return True
            except:
                # Fall back to OCR for scanned PDFs
                print(f"→ PDF appears scanned, using OCR: {file_path}")
                
                # Try Marker first (better for Hebrew and structure)
                if MARKER_AVAILABLE:
                    try:
                        import subprocess
                        import tempfile
                        import shutil
                        
                        # Use marker_single CLI from venv311
                        with tempfile.TemporaryDirectory() as tmpdir:
                            print("Converting with Marker (this may take a few minutes)...")
                            
                            # Call marker_single from venv311
                            result = subprocess.run(
                                ['./venv311/bin/marker_single', str(file_path), '--output_dir', tmpdir],
                                capture_output=True,
                                text=True,
                                cwd=Path(__file__).parent
                            )
                            
                            if result.returncode == 0:
                                # Find the generated markdown file
                                pdf_basename = Path(file_path).stem
                                marker_md_path = Path(tmpdir) / pdf_basename / f"{pdf_basename}.md"
                                
                                if marker_md_path.exists():
                                    # Copy to output location
                                    shutil.copy(marker_md_path, output_path)
                                    print(f"✓ Marker OCR: {file_path} -> {output_path}")
                                    return True
                                else:
                                    print(f"⚠️  Marker output not found at {marker_md_path}")
                            else:
                                print(f"⚠️  Marker failed: {result.stderr[:200]}")
                        
                    except Exception as e:
                        print(f"⚠️  Marker error: {e}, falling back to pytesseract")
                
                # Fall back to pytesseract
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(f"# {Path(file_path).name}\n\n")
                            f.write(f"*Source: {file_path}*\n\n")
                            f.write("---\n\n")
                            f.write(full_text)
                        
                        print(f"✓ Marker OCR'd PDF: {file_path} -> {output_path}")
                        return True
                    except Exception as marker_error:
                        print(f"⚠️  Marker failed ({marker_error}), falling back to pytesseract")
                
                # Fallback to pytesseract
                images = convert_from_path(file_path)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"# {Path(file_path).name}\n\n")
                    f.write(f"*Source: {file_path}*\n\n")
                    f.write("---\n\n")
                    
                    for i, image in enumerate(images):
                        if i > 0:
                            f.write(f"\n\n---\n\n## Page {i + 1}\n\n")
                        text = pytesseract.image_to_string(image, lang='heb+eng')
                        f.write(text)
                
                print(f"✓ Pytesseract OCR'd PDF: {file_path} -> {output_path}")
                return True
                
        except Exception as e:
            print(f"✗ Error processing PDF {file_path}: {e}")
            return False
    
    def process_file(self, file_path):
        """Process a file based on its type."""
        file_path = Path(file_path)
        
        if not file_path.is_file():
            return
        
        # Skip ignored files
        file_name_lower = file_path.name.lower()
        if file_name_lower in IGNORED_FILES:
            return
        
        # Skip files with ignored patterns
        if any(file_name_lower.endswith(pattern) for pattern in IGNORED_PATTERNS):
            return
        
        # Skip hidden files (starting with .)
        if file_path.name.startswith('.'):
            return
        
        suffix = file_path.suffix.lower()
        
        if suffix in SUPPORTED_TEXT_FORMATS:
            self.convert_text_to_md(file_path)
        elif suffix in SUPPORTED_IMAGE_FORMATS:
            self.convert_image_to_md(file_path)
        elif suffix in SUPPORTED_PDF:
            self.convert_pdf_to_md(file_path)
        else:
            print(f"⊘ Unsupported format: {file_path}")
    
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            print(f"\n→ New file detected: {event.src_path}")
            # Submit to thread pool for parallel processing
            file_path = event.src_path
            future = self.executor.submit(self._process_with_delay, file_path)
            self.pending_tasks[file_path] = future
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            print(f"\n→ File modified: {event.src_path}")
            # Submit to thread pool for parallel processing
            file_path = event.src_path
            future = self.executor.submit(self._process_with_delay, file_path)
            self.pending_tasks[file_path] = future
    
    def _process_with_delay(self, file_path):
        """Process file with delay to ensure it's fully written."""
        time.sleep(0.5)
        try:
            self.process_file(file_path)
        finally:
            # Clean up completed task
            self.pending_tasks.pop(file_path, None)
    
    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
        print("✓ All conversions completed")
    
    def on_deleted(self, event):
        """Handle file and directory deletion events."""
        if event.is_directory:
            # Handle directory deletion - remove corresponding output directory
            try:
                rel_path = Path(event.src_path).resolve().relative_to(self.watched_dir)
                output_dir_path = self.output_dir / rel_path
                
                if output_dir_path.exists():
                    shutil.rmtree(output_dir_path)
                    print(f"✓ Removed directory: {output_dir_path}")
                
                # Call ingest delete for all files in directory (if tracking)
                # For now, skip directory-level deletion from DB
            except Exception as e:
                print(f"✗ Error removing directory: {e}")
        else:
            # Handle file deletion
            try:
                source_path = Path(event.src_path).resolve()
                rel_path = self.get_relative_path(event.src_path)
                output_path = self.output_dir / rel_path.with_suffix('.md')
                
                if output_path.exists():
                    output_path.unlink()
                    print(f"✓ Removed: {output_path}")
                    
                    # Clean up empty parent directories
                    try:
                        parent = output_path.parent
                        while parent != self.output_dir and parent.exists():
                            if not any(parent.iterdir()):
                                parent.rmdir()
                                parent = parent.parent
                            else:
                                break
                    except Exception:
                        pass
                
            except Exception as e:
                print(f"✗ Error removing file: {e}")


def process_existing_files(handler, max_workers=4):
    """Sync files: process watched-dir files and remove orphaned md files."""
    print("\n=== Initial Sync: Scanning directories ===")
    
    # Step 1: Collect all source files and their expected output paths
    source_files = {}  # output_path -> source_path mapping
    files_to_process = []
    
    for root, _, files in os.walk(handler.watched_dir):
        for file in files:
            file_path = Path(root) / file
            # Pre-filter using the same logic as process_file
            file_name_lower = file_path.name.lower()
            if (not file_path.name.startswith('.') and 
                file_name_lower not in IGNORED_FILES and
                not any(file_name_lower.endswith(pattern) for pattern in IGNORED_PATTERNS)):
                
                # Check if this file type is supported
                suffix = file_path.suffix.lower()
                if suffix in (SUPPORTED_TEXT_FORMATS | SUPPORTED_IMAGE_FORMATS | SUPPORTED_PDF):
                    output_path = handler.get_output_path(file_path)
                    source_files[output_path] = file_path
                    files_to_process.append(file_path)
    
    # Step 2: Find and remove orphaned markdown files
    print(f"Scanning for orphaned files in {handler.output_dir}...")
    orphaned_count = 0
    
    if handler.output_dir.exists():
        for md_file in handler.output_dir.rglob('*.md'):
            if md_file not in source_files:
                try:
                    md_file.unlink()
                    orphaned_count += 1
                    print(f"🗑  Removed orphaned: {md_file.relative_to(handler.output_dir)}")
                except Exception as e:
                    print(f"✗ Error removing {md_file}: {e}")
        
        # Remove empty directories
        for dir_path in sorted(handler.output_dir.rglob('*'), reverse=True):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                except Exception:
                    pass
    
    if orphaned_count > 0:
        print(f"✓ Removed {orphaned_count} orphaned file(s)\n")
    
    # Step 3: Process source files in parallel
    total_files = len(files_to_process)
    if total_files == 0:
        print("No files to process.\n")
        return
    
    print(f"Found {total_files} file(s) to sync")
    print(f"Processing with {max_workers} workers...\n")
    
    processed = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(handler.process_file, file_path): file_path 
                          for file_path in files_to_process}
        
        # Process results as they complete
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                processed += 1
                # Show progress
                print(f"[{processed}/{total_files}] Synced: {file_path.name}")
            except Exception as e:
                failed += 1
                print(f"✗ Error processing {file_path}: {e}")
    
    print(f"\n=== Initial Sync Complete ===")
    print(f"✓ Synced: {processed}/{total_files}")
    if orphaned_count > 0:
        print(f"🗑  Removed: {orphaned_count} orphaned file(s)")
    if failed > 0:
        print(f"✗ Failed: {failed}")
    print()


def main():
    """Main function to start the file watcher."""
    print("=" * 60)
    print("RAG Pipeline File Watcher")
    print("=" * 60)
    print(f"Watching: {WATCHED_DIR.absolute()}")
    print(f"Output:   {OUTPUT_DIR.absolute()}")
    print("=" * 60)
    
    # Create directories if they don't exist
    WATCHED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create event handler and observer
    event_handler = FileConversionHandler(WATCHED_DIR, OUTPUT_DIR, max_workers=3)
    observer = Observer()
    observer.schedule(event_handler, str(WATCHED_DIR), recursive=True)
    
    # Process existing files first (with parallel processing)
    process_existing_files(event_handler, max_workers=4)
    
    # Start watching
    observer.start()
    print("\n👁  Watching for changes... (Press Ctrl+C to stop)\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping watcher...")
        observer.stop()
    
    observer.join()
    event_handler.cleanup()
    print("✓ Watcher stopped.\n")


if __name__ == "__main__":
    main()
