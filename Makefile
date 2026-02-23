.PHONY: pipeline front setup

# Create venvs with uv and install deps (idempotent: skips if .venv exists)
setup:
	@test -d services/watch/.venv || (cd services/watch && uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt)
	@test -d services/query/.venv || (cd services/query && uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt)
	@test -d services/ingest/.venv || (cd services/ingest && uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt)
	@test -d services/watch/pdf_converter/doclin-venv || (cd services/watch/pdf_converter && uv venv doclin-venv && uv pip install --python doclin-venv/bin/python -r requirements.txt)
	@test -d front/.venv || (cd front && uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt && uv pip install --python .venv/bin/python -r ../services/query/requirements.txt)

# Run the ingest + watch pipeline (run_pipeline.py)
pipeline: setup
	python3 run_pipeline.py

# Run the Gradio RAG query UI (front/start.py). Uses front/.venv if present.
front: setup
	@if [ -d front/.venv ]; then front/.venv/bin/python front/start.py; else python3 front/start.py; fi
