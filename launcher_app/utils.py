"""
Utility functions — file reading, subprocess execution.
========================================================
Shared helpers used by both CLI and GUI launchers.
"""

import os
import sys
import subprocess
import textwrap

from .config import BASE_DIR, SCRIPT_TIMEOUT


def get_file_docstring(filepath):
    """Extract the module-level docstring from a Python file.

    Reads the full file (needed for long docstrings with theory
    sections) and returns the content between the first pair of
    triple-quotes.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        for quote in ['"""', "'''"]:
            idx = content.find(quote)
            if idx != -1:
                end = content.find(quote, idx + 3)
                if end != -1:
                    return content[idx + 3:end].strip()
        return "No description available."
    except Exception:
        return "Could not read file."


def run_file(filepath):
    """Run a Python file as a subprocess (blocking, for CLI)."""
    print(f"\n{'='*60}")
    print(f"  Running: {os.path.basename(filepath)}")
    print(f"{'='*60}\n")
    try:
        result = subprocess.run(
            [sys.executable, filepath],
            cwd=os.path.dirname(filepath),
            timeout=SCRIPT_TIMEOUT
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"\n⚠️  Script timed out after {SCRIPT_TIMEOUT} seconds.")
        return -1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return -1
