.PHONY: help venv install clean run test ingest-venv ingest-install ingest-setup ingest ingest-stats ingest-clear query-venv query-install query-setup query clean-md clean-db deps setup

# Variables - Watch
VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# Variables - Ingest
INGEST_VENV = ingest-venv
INGEST_PYTHON = $(INGEST_VENV)/bin/python
INGEST_PIP = $(INGEST_VENV)/bin/pip

# Variables - Query
QUERY_VENV = query-venv
QUERY_PYTHON = $(QUERY_VENV)/bin/python
QUERY_PIP = $(QUERY_VENV)/bin/pip

# Default target
help:
	@echo "RAG Pipeline - Makefile Commands"
	@echo "========================================"
	@echo "Watch Pipeline (Lightweight):"
	@echo "  make venv       - Create watch virtual environment"
	@echo "  make install    - Install watch dependencies"
	@echo "  make setup      - Complete watch setup (venv + install)"
	@echo "  make run        - Run the file watcher"
	@echo "  make test       - Run watch test suite"
	@echo ""
	@echo "Ingest Pipeline (ML/Embeddings):"
	@echo "  make ingest-venv    - Create ingest virtual environment"
	@echo "  make ingest-install - Install ingest dependencies"
	@echo "  make ingest-setup   - Complete ingest setup"
	@echo "  make ingest         - Run ingestion pipeline"
	@echo "  make ingest-stats   - Show database statistics"
	@echo "  make ingest-clear   - Clear all data and history from database"
	@echo ""
	@echo "Query Pipeline (RAG with PydanticAI):"
	@echo "  make query-venv     - Create query virtual environment"
	@echo "  make query-install  - Install query dependencies"
	@echo "  make query-setup    - Complete query setup"
	@echo "  make query          - Run interactive query mode"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean      - Remove all virtual environments"
	@echo "  make clean-md   - Remove generated markdown files"
	@echo "  make clean-db   - Remove LanceDB database"
	@echo "  make deps       - Install system dependencies (macOS)"

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "✓ Virtual environment created"

# Install Python dependencies
install: venv
	@echo "Installing Python dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✓ Dependencies installed"

# Complete setup
setup: install
	@echo "✓ Setup complete! Run 'make run' to start the watcher"

# Install system dependencies (macOS)
deps:
	@echo "Installing system dependencies (macOS)..."
	@command -v brew >/dev/null 2>&1 || { echo "Error: Homebrew not found. Install from https://brew.sh"; exit 1; }
	brew install pandoc tesseract poppler tesseract-lang
	@echo "✓ System dependencies installed (including Hebrew language support)"

# Run the watcher
run: venv
	@echo "Starting file watcher..."
	$(PYTHON) watch.py

# Sync watched-dir to md-content once (no watching)
sync-once: venv
	@echo "Syncing watched-dir to md-content..."
	$(PYTHON) -c "from watch import FileConversionHandler, process_existing_files, WATCHED_DIR, OUTPUT_DIR; handler = FileConversionHandler(WATCHED_DIR, OUTPUT_DIR); process_existing_files(handler); print('✓ Sync complete')"

# Run tests
test: venv
	@echo "Running test suite..."
	$(PYTHON) test_watch.py

# Create ingest virtual environment
ingest-venv:
	@echo "Creating ingest virtual environment..."
	python3 -m venv $(INGEST_VENV)
	@echo "✓ Ingest virtual environment created"

# Install ingest dependencies
ingest-install: ingest-venv
	@echo "Installing ingest dependencies (this may take a few minutes)..."
	$(INGEST_PIP) install --upgrade pip
	$(INGEST_PIP) install -r requirements-ingest.txt
	@echo "✓ Ingest dependencies installed"

# Complete ingest setup
ingest-setup: ingest-install
	@echo "✓ Ingest setup complete! Run 'make ingest' to process documents"

# Run ingestion pipeline (syncs watched-dir first)
ingest: ingest-venv venv sync-once
	@echo "Running ingestion pipeline..."
	$(INGEST_PYTHON) ingest.py

# Show database statistics
ingest-stats: ingest-venv
	@echo "Database Statistics:"
	$(INGEST_PYTHON) ingest.py --stats

# Clear database history and data
ingest-clear: ingest-venv
	@echo "Clearing database..."
	$(INGEST_PYTHON) ingest.py --clear

# === Query Pipeline (PydanticAI + Ollama) ===

# Create query virtual environment
query-venv:
	@echo "Creating query virtual environment..."
	python3 -m venv $(QUERY_VENV)
	@echo "✓ Query virtual environment created"

# Install query dependencies
query-install: query-venv
	@echo "Installing query dependencies..."
	$(QUERY_PIP) install --upgrade pip
	$(QUERY_PIP) install -r requirements-query.txt
	@echo "✓ Query dependencies installed"

# Complete query setup
query-setup: query-install
	@echo "✓ Query setup complete!"
	@echo ""
	@echo "Make sure Ollama is installed and running:"
	@echo "  1. Install: https://ollama.ai"
	@echo "  2. Pull model: ollama pull llama3.2"
	@echo "  3. Start Ollama service"
	@echo ""
	@echo "Then run 'make query' to start querying your documents"

# Run query in interactive mode
query: query-install
	@echo "Starting RAG Query Assistant..."
	$(QUERY_PYTHON) query.py --interactive

# Clean up
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV)
	rm -rf $(INGEST_VENV)
	rm -rf $(QUERY_VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "✓ Cleanup complete"

# Clean generated markdown files
clean-md:
	@echo "Removing generated markdown files..."
	rm -rf md-content/*
	@echo "✓ Markdown files removed"

# Clean LanceDB database
clean-db:
	@echo "Removing LanceDB database..."
	rm -rf lancedb/
	@echo "✓ Database removed"
