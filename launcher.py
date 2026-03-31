"""
Numerical Methods Launcher (Compatibility Wrapper)
=================================================
Backward-compatible entry point that delegates to the modular
`launcher_app` package.

Usage:
    python launcher.py          # auto-detect mode (GUI if available)
    python launcher.py --cli    # force terminal mode
    python launcher.py --gui    # force GUI mode

Public symbols are re-exported for compatibility with older imports.
"""

from launcher_app.catalog import CATEGORIES, LEARNING_PATH
from launcher_app.equations import EQUATIONS, is_math_line
from launcher_app.utils import get_file_docstring, run_file
from launcher_app.cli import cli_launcher
from launcher_app.gui import gui_launcher
from launcher_app.main import main
from launcher_app.config import BASE_DIR

__all__ = [
    "BASE_DIR",
    "CATEGORIES",
    "LEARNING_PATH",
    "EQUATIONS",
    "is_math_line",
    "get_file_docstring",
    "run_file",
    "cli_launcher",
    "gui_launcher",
    "main",
]


if __name__ == "__main__":
    main()
