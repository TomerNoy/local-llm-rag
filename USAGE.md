# RAG Query System - Usage Guide

## Overview
This RAG (Retrieval Augmented Generation) system monitors directories for document changes, ingests them into a LanceDB vector database, and provides an AI-powered query interface.

## System Components

### 1. watch.py (File Monitoring)
Monitors `watched-dir/` for file changes and triggers ingestion.

**Usage:**
```bash
python watch.py
```

### 2. ingest.py (Document Ingestion)
Processes documents, creates embeddings, and stores in LanceDB with hash-based change detection.

**Supported formats:** PDF (with OCR), RTF, TXT, DOCX, MD

**Usage:**
```bash
# Process all files in watched-dir
python ingest.py

# Process specific file
python ingest.py path/to/file.pdf
```

### 3. query.py (AI Query Interface)
PydanticAI-powered assistant using Ollama's qwen2.5:7b model with 10 registered tools.

**Usage:**
```bash
# Interactive mode
python query.py

# Single query
python query.py "list all files"
```

## Available Query Types

### File Listing & Discovery
- `"list all files"` - Shows all indexed files with metadata
- `"find files with name X"` - Search by filename
- `"show files from folder X"` - Filter by folder/tag
- `"list pdf files"` - Filter by type (pdf, rtf, txt, docx, md)
- `"show files from last 7 days"` - Filter by recency

### Content Search & Retrieval
- `"search for text about X"` - Semantic search across content
- `"summarize all files"` - Quick preview of all file contents
- `"summarize the text1 file"` - Get specific file content
- `"get content of X file"` - Retrieve all chunks from a file

### Statistics
- `"how many files do we have?"` - File count
- `"database stats"` - Comprehensive statistics

## Features

### Hebrew & Unicode Support
- ✅ Hebrew filenames preserved correctly ("מכבי.pdf")
- ✅ UTF-8 storage in database
- ✅ Pre-formatted output bypasses LLM corruption

### Anti-Hallucination Measures
- All responses require tool calls to database
- System prompt forbids fabrication
- Content retrieval tools verify information

### Batch Operations
- `summarize_all_files()` tool handles multiple files
- No manual looping by LLM (prevents Hebrew corruption)
- Pre-formatted output with all filenames preserved

### Hash-Based Change Detection
- Only reprocesses files when content changes
- Efficient incremental updates
- Preserves existing embeddings

## Database Schema

Each document chunk contains:
- `chunk_id`: Unique identifier
- `file_name`: Original filename (UTF-8)
- `source_path`: Full path to source file
- `content`: Text content of chunk
- `chunk_index`: Position in document
- `file_type`: Document type (pdf, rtf, etc.)
- `tags`: Folder hierarchy as tags
- `timestamp`: Last modified time
- `vector`: 384-dim embedding (all-MiniLM-L6-v2)

## Technical Stack

- **LanceDB 0.27.1**: Vector database
- **PydanticAI 0.8.1**: Agent framework
- **Ollama qwen2.5:7b**: LLM via OpenAI-compatible API
- **Sentence-transformers**: all-MiniLM-L6-v2 embeddings (384 dim)
- **PyMuPDF + Tesseract**: PDF processing with OCR
- **Watchdog**: File system monitoring

## Configuration

### Environment Variables
```bash
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
```

### Database Location
- Database: `lancedb_data/documents`
- Embeddings: `lancedb_data/documents.lance/`

## Testing

Run comprehensive test suite:
```bash
pytest test_ingest.py -v
```

12/12 tests passing including:
- Text extraction
- Data validation
- Hash-based change detection
- Chunking strategies
- Error handling

## Known Limitations

1. **Hebrew in LLM text generation**: Qwen2.5:7b occasionally corrupts Hebrew in generated text (not in tool output). Mitigated by pre-formatted strings.

2. **Model behavior**: Qwen2.5:7b sometimes adds Chinese explanations. System prompt configured to minimize this.

3. **OCR accuracy**: PDF OCR depends on image quality.

## Example Queries

```bash
# Basic operations
python query.py "list all files"
python query.py "how many files do we have?"

# Content search
python query.py "search for legal documents"
python query.py "find text about contracts"

# Batch operations  
python query.py "summarize all files"
python query.py "summarize each file"

# Filtered queries
python query.py "show pdf files"
python query.py "list files from test folder"
python query.py "show files from last 3 days"

# Specific retrieval
python query.py "get content of text1 file"
python query.py "summarize the legal_1 file"
```

## Architecture Notes

### Why Pre-Formatted Output?
Tools return pre-formatted strings (not dicts) to bypass LLM text generation. This preserves Hebrew filenames and special characters that qwen2.5:7b might corrupt.

### Why Batch Tool?
The `summarize_all_files()` tool handles loops in Python rather than letting the LLM iterate. This prevents Hebrew corruption when filenames are passed as tool arguments.

### Why OpenAI-Compatible API?
Using `openai:qwen2.5:7b` with Ollama's OpenAI-compatible endpoint (`/v1`) enables proper tool execution. Direct `ollama:` prefix is unsupported by PydanticAI.

## Current Database Status

- **3 files indexed**:
  - מכבי.pdf (Hebrew filename, 4 chunks)
  - legal_1.rtf (test folder, 3 chunks)
  - text1.rtf (test/ok folder, 1 chunk)

- **Total chunks**: 8
- **Embedding dimensions**: 384
- **Model**: all-MiniLM-L6-v2
