"""
Gradio UI for RAG query. Run from project root or front/ with project root on path.
Uses the same agent as services/query/query.py.
"""
import sys
from pathlib import Path

# Project root so we can import the query service
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.query.query import agent, db, check_llm_reachable
import gradio as gr


def answer(message, history):
    """Chat callback: run agent on the user message and return the response."""
    if not message or not message.strip():
        return ""
    if db.table is None:
        return "No database found. Run ingest first to index your documents."
    ok, err = check_llm_reachable()
    if not ok:
        return err
    try:
        result = agent.run_sync(message.strip())
        return result.output or ""
    except Exception as e:
        return f"Error: {e}"


# Chat interface: minimal UI, one question at a time (no conversation history sent to agent)
demo = gr.ChatInterface(
    fn=answer,
    title="RAG Query",
    description=f"Ask questions about your documents. ({db.count_files()} files indexed.)",
)

if __name__ == "__main__":
    demo.launch()
