#!/usr/bin/env python3
"""
Pipeline orchestrator for local-llm-rag.

Phase 1: Initial sync (watched-dir -> markdown) + ingest (markdown -> vector DB)
Phase 2: Continuous watch for file changes, ingest triggered on change.

No external dependencies - uses only Python stdlib.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Load shared config
_config = json.loads((PROJECT_ROOT / "config.json").read_text())

# Seconds to wait after last detected change before running ingest.
# Batches rapid changes (e.g. dropping 10 files at once) into a single ingest run.
DEBOUNCE_SECONDS = _config.get("ingest_debounce_seconds", 5)

# Directories that must exist before running
REQUIRED_DIRS = [
    PROJECT_ROOT / _config["watched_dir"],
    PROJECT_ROOT / _config["md_content_dir"],
    PROJECT_ROOT / _config["lancedb_dir"],
]

# One-shot sync step
SYNC_STEP = {
    "name": "Watch Sync",
    "description": "Convert watched-dir files to markdown",
    ".venv": PROJECT_ROOT / "services" / "watch" / ".venv",
    "cwd": PROJECT_ROOT / "services" / "watch",
    "cmd_args": [
        "-c",
        (
            "from watch import FileConversionHandler, process_existing_files, "
            "WATCHED_DIR, MD_CONTENT_DIR; "
            "handler = FileConversionHandler(WATCHED_DIR, MD_CONTENT_DIR); "
            "process_existing_files(handler); "
            "handler.cleanup()"
        ),
    ],
}

# Ingest step
INGEST_STEP = {
    "name": "Ingest",
    "description": "Chunk, embed, and store in LanceDB",
    ".venv": PROJECT_ROOT / "services" / "ingest" / ".venv",
    "cwd": PROJECT_ROOT / "services" / "ingest",
    "cmd_args": ["ingest.py"],
}

# Continuous watch daemon
WATCH_DAEMON = {
    "name": "Watch Daemon",
    ".venv": PROJECT_ROOT / "services" / "watch" / ".venv",
    "cwd": PROJECT_ROOT / "services" / "watch",
    "cmd_args": ["watch.py"],
}

# Sentinel file touched by watch daemon when it processes files; we poll this instead of
# piping stdout so the watch process never blocks on a full pipe (which can drop events).
WATCH_SIGNAL_FILE = PROJECT_ROOT / "storage" / ".watch-signal"
POLL_INTERVAL_SECONDS = 1.5


def get_python(venv_path: Path) -> Path:
    return venv_path / "bin" / "python"


def ensure_dirs():
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def check_venv(step: dict) -> bool:
    python = get_python(step[".venv"])
    if not python.exists():
        print(f"  ERROR: .venv not found at {step['.venv']}")
        print(f"  Fix:   cd {step['cwd']} && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt")
        return False
    return True


def run_step(step: dict, prefix: str = "") -> bool:
    """Run a pipeline step, streaming output in real-time. Returns True on success."""
    if not check_venv(step):
        return False

    cmd = [str(get_python(step[".venv"]))] + step["cmd_args"]

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(step["cwd"]),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(f"  {prefix}{line}", end="")

        process.wait()

        if process.returncode != 0:
            print(f"  {prefix}ERROR: process exited with code {process.returncode}")
            return False

        return True

    except Exception as e:
        print(f"  {prefix}ERROR: {e}")
        return False


def start_watch_daemon(change_event: threading.Event) -> subprocess.Popen:
    """Start the watch daemon. Change detection via sentinel file (no stdout pipe)."""
    cmd = [str(get_python(WATCH_DAEMON[".venv"]))] + WATCH_DAEMON["cmd_args"]

    # Do not pipe stdout/stderr: when piped, the watch process uses block-buffered stdout.
    # Prints run in the Observer's event thread; if the pipe fills, that thread blocks and
    # filesystem events can be missed. Let watch write to the terminal like when run directly.
    process = subprocess.Popen(
        cmd,
        cwd=str(WATCH_DAEMON["cwd"]),
        stdout=None,
        stderr=None,
    )

    last_mtime = [0.0]  # mutable so the thread can update it

    def poll_signal():
        while True:
            time.sleep(POLL_INTERVAL_SECONDS)
            try:
                if WATCH_SIGNAL_FILE.exists():
                    m = os.path.getmtime(WATCH_SIGNAL_FILE)
                    if m > last_mtime[0]:
                        last_mtime[0] = m
                        change_event.set()
            except Exception:
                pass

    thread = threading.Thread(target=poll_signal, daemon=True)
    thread.start()

    return process


def main():
    print("=" * 60)
    print("Local LLM RAG Pipeline")
    print("=" * 60)

    ensure_dirs()

    # --- Phase 1: Initial sync + ingest ---

    print("\n--- Phase 1: Initial sync + ingest ---\n")

    for step in [SYNC_STEP, INGEST_STEP]:
        print(f"[{step['name']}] {step['description']}")
        print("-" * 50)

        start = time.time()
        success = run_step(step)
        elapsed = time.time() - start

        status = "OK" if success else "FAILED"
        print(f"\n  => {status} ({elapsed:.1f}s)\n")

        if not success:
            print(f"Pipeline stopped: '{step['name']}' failed.")
            sys.exit(1)

    # --- Phase 2: Continuous watch, ingest on change ---

    print("--- Phase 2: Watching for changes ---\n")

    if not check_venv(WATCH_DAEMON):
        sys.exit(1)

    change_event = threading.Event()
    watch_proc = start_watch_daemon(change_event)

    print(f"  File watcher started (pid {watch_proc.pid})")
    print(f"  Ingest triggers on file changes (debounce: {DEBOUNCE_SECONDS}s)")
    print(f"  Press Ctrl+C to stop\n")

    try:
        while True:
            # Block until watch detects a change
            change_event.wait()

            # Check if watch daemon is still alive
            if watch_proc.poll() is not None:
                print(f"\n  ERROR: Watch daemon exited unexpectedly (code {watch_proc.returncode})")
                sys.exit(1)

            # Debounce: wait for rapid changes to settle
            time.sleep(DEBOUNCE_SECONDS)
            change_event.clear()

            print(f"\n  [ingest] Change detected, running ingest...")
            print("  " + "-" * 48)
            success = run_step(INGEST_STEP, prefix="[ingest] ")
            if not success:
                print("  [ingest] WARNING: ingest failed, will retry on next change")

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        watch_proc.terminate()
        try:
            watch_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            watch_proc.kill()
        print("Pipeline stopped.")


if __name__ == "__main__":
    main()
