.PHONY: pipeline front setup

install:
	uv sync --all-packages
# Create venvs with uv and install deps (idempotent: skips if .venv exists)
setup: install
	@test -d services/watch/.venv || (cd services/watch && uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt)
	@test -d services/query/.venv || (cd services/query && uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt)
	@test -d services/ingest/.venv || (cd services/ingest && uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt)
	@test -d services/watch/pdf_converter/doclin-venv || (cd services/watch/pdf_converter && uv venv doclin-venv && uv pip install --python doclin-venv/bin/python -r requirements.txt)
	@test -d front/.venv || (cd front && uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt)


# Run the ingest + watch pipeline (run_pipeline.py)
pipeline:
	uv run run_pipeline.py

# Run the Gradio RAG query UI (front/start.py). Uses front/.venv if present.
front:
	cd front && uv run start.py
