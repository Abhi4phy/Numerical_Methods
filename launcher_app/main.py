"""
Application entry point — mode detection and dispatch.
========================================================
Determines whether to launch CLI or GUI based on
command-line flags and display availability.
"""

import sys


def main():
    """Detect mode and launch appropriate interface."""
    mode = None
    if "--cli" in sys.argv:
        mode = "cli"
    elif "--gui" in sys.argv:
        mode = "gui"

    if mode == "cli":
        from .cli import cli_launcher
        cli_launcher()
    elif mode == "gui":
        from .gui import gui_launcher
        gui_launcher()
    else:
        # Auto-detect: try GUI, fall back to CLI
        try:
            import tkinter
            test_root = tkinter.Tk()
            test_root.withdraw()
            test_root.destroy()
            from .gui import gui_launcher
            gui_launcher()
        except Exception:
            print("  GUI not available, starting terminal mode...\n")
            from .cli import cli_launcher
            cli_launcher()
