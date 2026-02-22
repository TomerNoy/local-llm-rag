# Local LLM RAG

A local, read-only LLM service over your personal files with RAG support. Drop files into a watched folder, and query them with natural language using a local LLM.

## How It Works

```
watched-dir/          services/watch/       storage/md-content/       services/ingest/       storage/lancedb/       services/query/
  your-files/ ──────► watch.py ──────────► markdown files ──────────► ingest.py ──────────► vector DB ──────────► query.py + LLM
  (pdf,txt,img)       (convert to md)                                 (chunk & embed)                              (RAG search)
```

## Project Structure

```
local-llm-rag/
├── services/
│   ├── watch/          # File watcher: monitors watched-dir, converts to markdown
│   │   ├── watch.py
│   │   ├── watcher_config.py
│   │   ├── requirements.txt
│   │   └── pdf_converter/
│   │       ├── pdf_to_markdown.py
│   │       └── requirements.txt
│   ├── ingest/         # Ingestion: chunks markdown, generates embeddings, stores in LanceDB
│   │   ├── ingest.py
│   │   └── requirements.txt
│   └── query/          # Query: PydanticAI agent with RAG tools over LanceDB
│       ├── query.py
│       └── requirements.txt
├── storage/            # Runtime data (gitignored)
│   ├── md-content/     # Generated markdown files
│   └── lancedb/        # Vector database
└── watched-dir/        # Drop your files here (gitignored)
```

Each service has its own virtual environment and dependencies (avoids cross-service dependency conflicts).

**IDE:** When editing a service, select that service’s venv as the Python interpreter (e.g. `services/watch/venv/bin/python` for `watch.py`) so imports resolve correctly.

## Setup

### System Dependencies (macOS)

```bash
brew install pandoc tesseract poppler tesseract-lang
```

### 1. Watch (file monitoring + conversion)

```bash
cd services/watch
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python watch.py
```

PDF support requires an additional venv for Docling (heavy ML dependencies):

```bash
cd services/watch/pdf_converter
python3 -m venv doclin-venv && source doclin-venv/bin/activate
pip install -r requirements.txt
```

### 2. Ingest (embedding + vector DB)

```bash
cd services/ingest
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python ingest.py
```

Options: `--force` (re-ingest all), `--stats`, `--compact`, `--clear`, `--search "query"`

### 3. Query (LLM + RAG)

Requires [LM Studio](https://lmstudio.ai/) running locally with a model loaded (e.g. qwen3-14b).

```bash
cd services/query
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Single query
python query.py "how many files do I have?"

# Interactive mode
python query.py --interactive
```

The LLM endpoint is configurable via environment variables:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:1234/v1   # default: LM Studio
export OPENAI_API_KEY=lm-studio
```

## Pipeline (all-in-one)

To run initial sync + ingest and then keep watching for changes (no need to start watch and ingest separately):

```bash
# From project root. Requires watch + ingest venvs already set up (see above).
python run_pipeline.py
```

Phase 1: one-shot sync (watched-dir → markdown) and ingest (markdown → vector DB).  
Phase 2: file watcher runs; ingest re-runs automatically after changes (debounce ~5s).  
Stop with Ctrl+C.

## Features

- **Format support**: PDF (digital + scanned/OCR), TXT, DOCX, RTF, HTML, images (OCR)
- **Hebrew + multilingual**: Uses `paraphrase-multilingual-MiniLM-L12-v2` embeddings
- **Incremental sync**: Hash-based change detection, only re-processes modified files
- **RAG tools**: Semantic search, file listing, content retrieval, stats, summarization
