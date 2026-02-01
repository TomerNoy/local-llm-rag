# RAG Pipeline File Watcher

A Python file watcher that monitors `watched-dir` and automatically converts documents to markdown format in `md-content/`, preserving folder structure for LanceDB ingestion.

## Features

- **Real-time Monitoring**: Watches `watched-dir` for new, modified, or deleted files
- **Format Support**: 
  - Text documents: `.txt`, `.doc`, `.docx`, `.rtf`, `.html`, `.odt`
  - Images: `.jpg`, `.jpeg`, `.png`, `.tiff`, `.bmp` (with OCR)
  - PDFs: Both text-based and scanned documents (with OCR fallback)
- **Structure Preservation**: Maintains original folder structure in output
- **OCR Support**: Automatically applies OCR to images and scanned PDFs

## Quick Start

```bash
# Complete setup (macOS)
make deps    # Install system dependencies
make setup   # Create venv and install Python packages

# Run the watcher
make run
```

## Prerequisites

### System Dependencies

**macOS:**
```bash
make deps
# or manually:
brew install pandoc tesseract poppler tesseract-lang
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install pandoc tesseract-ocr tesseract-ocr-heb poppler-utils
```

### Python Dependencies

```bash
make setup
# or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. **Start the watcher:**
   ```bash
   make run
   ```

2. **Add files to `watched-dir/`**: The script will automatically detect and convert them

3. **Stop the watcher**: Press `Ctrl+C`

## Makefile Commands

- `make help` - Show all available commands
- `make venv` - Create virtual environment
- `make install` - Install Python dependencies
- `make setup` - Complete setup (venv + install)
- `make run` - Run the file watcher
- `make deps` - Install system dependencies (macOS)
- `make clean` - Remove venv and cache files
- `make clean-md` - Remove all generated markdown files

## How It Works

1. On startup, processes all existing files in `watched-dir`
2. Monitors for file changes (create, modify, delete)
3. Converts documents to markdown format
4. Saves to `md-content/` maintaining the same folder structure
5. Removes corresponding markdown files when source files are deleted

## Output Format

- Text-based documents: Converted using Pandoc
- Images/Scanned PDFs: OCR applied, output includes:
  - Original filename as heading
  - Source path metadata
  - Extracted text content

## For LanceDB Integration

The markdown files in `md-content/` are structured to be chunked and ingested into LanceDB for RAG (Retrieval-Augmented Generation) applications.

## Notes

- Hidden files (starting with `.`) are ignored
- Files are processed with a 0.5s delay to ensure complete writing
- Unsupported file formats are logged but not processed
