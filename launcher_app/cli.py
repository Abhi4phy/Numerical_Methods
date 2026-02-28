"""
CLI Launcher — terminal-based interactive menu.
=================================================
Provides keyboard-driven navigation through categories,
a 7-stage learning path, keyword search, and direct
file execution.
"""

import os
import textwrap

from .config import BASE_DIR
from .catalog import CATEGORIES, LEARNING_PATH
from .utils import get_file_docstring, run_file


# ────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def cli_header():
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║        NUMERICAL METHODS FOR PHYSICS — LAUNCHER          ║")
    print("║        57 methods across 16 categories                   ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()


# ────────────────────────────────────────────────────────
# Menus
# ────────────────────────────────────────────────────────

def cli_main_menu():
    """Show main menu and return choice."""
    cli_header()
    print("  Choose a mode:\n")
    print("    [B]  Browse by Category")
    print("    [L]  Learning Path (recommended order)")
    print("    [S]  Search methods")
    print("    [R]  Run a specific file")
    print("    [Q]  Quit")
    print()
    return input("  → ").strip().lower()


def cli_browse():
    """Browse methods by category."""
    while True:
        clear_screen()
        cli_header()
        print("  ── Categories ──\n")
        for i, cat in enumerate(CATEGORIES):
            n = len(cat["files"])
            print(f"    {i+1:2d}. {cat['icon']}  {cat['name']:30s} ({n} files)")
        print(f"\n    {'':2s}  [B] Back to main menu")
        print()

        choice = input("  Select category (1-16) → ").strip().lower()
        if choice == 'b' or choice == '':
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(CATEGORIES):
                cli_category(CATEGORIES[idx])
        except ValueError:
            pass


def cli_category(cat):
    """Show files in a category."""
    while True:
        clear_screen()
        cli_header()
        print(f"  ── {cat['icon']}  {cat['name']} ──\n")

        for i, (fname, title, desc) in enumerate(cat["files"]):
            filepath = os.path.join(BASE_DIR, cat["folder"], fname)
            exists = "✓" if os.path.exists(filepath) else "✗"
            print(f"    {i+1:2d}. [{exists}] {title:35s} — {desc}")

        print(f"\n    {'':2s}  [B] Back to categories")
        print()

        choice = input("  Select file (number) or [V]iew/[R]un → ").strip().lower()
        if choice == 'b' or choice == '':
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(cat["files"]):
                fname, title, desc = cat["files"][idx]
                filepath = os.path.join(BASE_DIR, cat["folder"], fname)
                cli_file_action(filepath, title, desc)
        except ValueError:
            pass


def cli_file_action(filepath, title, desc):
    """Show file actions."""
    while True:
        clear_screen()
        cli_header()
        print(f"  ── {title} ──")
        print(f"  {desc}\n")
        print(f"  File: {os.path.relpath(filepath, BASE_DIR)}")

        if os.path.exists(filepath):
            print("  Status: ✓ Available")
        else:
            print("  Status: ✗ File not found")
            input("\n  Press Enter to go back...")
            return

        print(f"\n    [V]  View description (docstring)")
        print(f"    [R]  Run the demo")
        print(f"    [O]  Open in editor")
        print(f"    [B]  Back")
        print()

        choice = input("  → ").strip().lower()

        if choice == 'b' or choice == '':
            return
        elif choice == 'v':
            clear_screen()
            cli_header()
            print(f"  ── {title} ── Description ──\n")
            docstring = get_file_docstring(filepath)
            for line in docstring.split('\n'):
                if len(line) > 76:
                    for wrapped in textwrap.wrap(line, 76):
                        print(f"  {wrapped}")
                else:
                    print(f"  {line}")
            print()
            input("  Press Enter to continue...")
        elif choice == 'r':
            run_file(filepath)
            print()
            input("  Press Enter to continue...")
        elif choice == 'o':
            try:
                import subprocess as sp
                if os.name == 'nt':
                    os.startfile(filepath)
                else:
                    sp.Popen(['xdg-open', filepath])
            except Exception:
                print(f"  Could not open file. Path: {filepath}")
                input("  Press Enter to continue...")


def cli_learning_path():
    """Show the recommended learning path."""
    while True:
        clear_screen()
        cli_header()
        print("  ── 📚 Recommended Learning Path ──\n")

        file_num = 0
        stages = []
        for stage_name, files in LEARNING_PATH:
            stages.append((stage_name, file_num, len(files)))
            file_num += len(files)

        for i, (stage_name, start, count) in enumerate(stages):
            print(f"    {i+1}. {stage_name:40s} ({count} methods)")

        print(f"\n    [B] Back")
        print()

        choice = input("  Select stage (1-7) → ").strip().lower()
        if choice == 'b' or choice == '':
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(LEARNING_PATH):
                cli_learning_stage(LEARNING_PATH[idx])
        except ValueError:
            pass


def cli_learning_stage(stage_data):
    """Show files in a learning stage."""
    stage_name, files = stage_data
    while True:
        clear_screen()
        cli_header()
        print(f"  ── 📚 {stage_name} ──\n")

        for i, (folder, fname, desc) in enumerate(files):
            filepath = os.path.join(BASE_DIR, folder, fname)
            exists = "✓" if os.path.exists(filepath) else "✗"
            display_name = fname.replace('.py', '').replace('_', ' ').title()
            print(f"    {i+1:2d}. [{exists}] {display_name:35s} — {desc}")

        print(f"\n    [B] Back")
        print()

        choice = input("  Select (number to view/run) → ").strip().lower()
        if choice == 'b' or choice == '':
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                folder, fname, desc = files[idx]
                filepath = os.path.join(BASE_DIR, folder, fname)
                title = fname.replace('.py', '').replace('_', ' ').title()
                cli_file_action(filepath, title, desc)
        except ValueError:
            pass


def cli_search():
    """Search for methods by keyword."""
    clear_screen()
    cli_header()
    print("  ── 🔍 Search Methods ──\n")
    query = input("  Search: ").strip().lower()

    if not query:
        return

    results = []
    for cat in CATEGORIES:
        for fname, title, desc in cat["files"]:
            searchable = f"{title} {desc} {fname}".lower()
            if query in searchable:
                results.append((cat, fname, title, desc))

    if not results:
        for cat in CATEGORIES:
            for fname, title, desc in cat["files"]:
                filepath = os.path.join(BASE_DIR, cat["folder"], fname)
                if os.path.exists(filepath):
                    docstring = get_file_docstring(filepath).lower()
                    if query in docstring:
                        results.append((cat, fname, title, desc))

    if not results:
        print(f"\n  No results for '{query}'.")
        input("  Press Enter to continue...")
        return

    print(f"\n  Found {len(results)} result(s):\n")
    for i, (cat, fname, title, desc) in enumerate(results):
        print(f"    {i+1:2d}. [{cat['name']}] {title:30s} — {desc}")

    print(f"\n    [B] Back")
    print()

    choice = input("  Select (number) → ").strip().lower()
    if choice == 'b' or choice == '':
        return

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(results):
            cat, fname, title, desc = results[idx]
            filepath = os.path.join(BASE_DIR, cat["folder"], fname)
            cli_file_action(filepath, title, desc)
    except ValueError:
        pass


def cli_run_direct():
    """Run a file by path."""
    clear_screen()
    cli_header()
    print("  ── Run File ──\n")
    print("  Enter relative path (e.g., 01_Linear_Algebra/svd.py)")
    print()
    path = input("  Path: ").strip()

    if not path:
        return

    import sys
    filepath = os.path.join(BASE_DIR, path)
    if os.path.exists(filepath):
        run_file(filepath)
        input("\n  Press Enter to continue...")
    else:
        print(f"\n  File not found: {filepath}")
        input("  Press Enter to continue...")


# ────────────────────────────────────────────────────────
# Main CLI event loop
# ────────────────────────────────────────────────────────

def cli_launcher():
    """Main CLI event loop."""
    while True:
        clear_screen()
        choice = cli_main_menu()

        if choice == 'b':
            cli_browse()
        elif choice == 'l':
            cli_learning_path()
        elif choice == 's':
            cli_search()
        elif choice == 'r':
            cli_run_direct()
        elif choice in ('q', 'quit', 'exit'):
            print("\n  Goodbye! Happy computing 🚀\n")
            break
