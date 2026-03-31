"""
Utility functions — file reading, subprocess execution.
========================================================
Shared helpers used by both CLI and GUI launchers.
"""

import os
import sys
import subprocess

from .config import SCRIPT_TIMEOUT


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


def parse_docstring_section(docstring, section_name):
    """Extract a named section from docstring.
    
    Looks for section headers like:
    ## Prerequisites
    ## Real-World Applications
    ## Common Pitfalls
    ## Complexity Analysis
    ## Related Methods
    
    Returns list of lines in that section, or empty list if not found.
    """
    lines = docstring.split('\n')
    section_marker = f"## {section_name}"
    
    in_section = False
    result = []
    
    for line in lines:
        if section_marker in line:
            in_section = True
            continue
        
        if in_section:
            # End section if we hit another ## header
            if line.strip().startswith("##") and section_marker not in line:
                break
            # Skip empty lines at section start
            if result or line.strip():
                result.append(line)
    
    # Clean up trailing empty lines
    while result and not result[-1].strip():
        result.pop()
    
    return result


def extract_metadata_from_docstring(docstring):
    """Extract all metadata sections from a docstring.
    
    Returns dict with keys: prerequisites, applications, pitfalls, 
    complexity, related_methods (each as list of strings).
    """
    return {
        "prerequisites": parse_docstring_section(docstring, "Prerequisites"),
        "applications": parse_docstring_section(docstring, "Real-World Applications"),
        "pitfalls": parse_docstring_section(docstring, "Common Pitfalls"),
        "complexity": parse_docstring_section(docstring, "Complexity Analysis"),
        "related_methods": parse_docstring_section(docstring, "Related Methods"),
    }


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


def run_file_with_capture(filepath, timeout=SCRIPT_TIMEOUT):
    """Run a Python file and capture stdout/stderr.
    
    Returns tuple: (stdout_str, stderr_str, return_code)
    """
    try:
        result = subprocess.run(
            [sys.executable, filepath],
            cwd=os.path.dirname(filepath),
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", f"⚠️ Script timed out after {timeout} seconds.", -1
    except Exception as e:
        return "", f"❌ Error running file: {e}", -1


def get_source_code(filepath, exclude_docstring=True):
    """Read source code from a Python file.
    
    Parameters:
        filepath: Path to .py file
        exclude_docstring: If True, strip module docstring
    
    Returns:
        Full source code as string
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if exclude_docstring:
            # Remove leading docstring if present
            for quote in ['"""', "'''"]:
                idx = content.find(quote)
                if idx == 0:  # Starts with docstring
                    end = content.find(quote, 3)
                    if end != -1:
                        content = content[end + 3:].lstrip('\n')
                        break
        
        return content
    except Exception as e:
        return f"# Error reading file: {e}"


def extract_key_function(filepath, func_name=""):
    """Extract main function/class implementation from source.
    
    Looks for functions/classes and extracts their complete definition.
    If func_name is empty, extracts the first major function/class after docstring.
    
    Returns:
        (function_name, source_code, start_line, end_line) or None if not found
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip docstring
        in_docstring = False
        start_idx = 0
        for i, line in enumerate(lines):
            if i == 0 and (line.strip().startswith('"""') or line.strip().startswith("'''")):
                in_docstring = True
                quote = '"""' if '"""' in line else "'''"
            if in_docstring:
                if quote in line and i > 0:
                    in_docstring = False
                    start_idx = i + 1
                    break
        
        # Find function/class definitions after docstring
        search_start = start_idx
        target_line = -1
        
        for i in range(search_start, len(lines)):
            stripped = lines[i].strip()
            if stripped.startswith("def "):
                # Skip private functions
                if not stripped[4:].startswith("_"):
                    target_line = i
                    break
            elif stripped.startswith("class "):
                target_line = i
                break
        
        if target_line == -1:
            return None
        
        # Extract function/class body (balanced indentation)
        func_name = lines[target_line].split('(')[0].replace('def ', '').replace('class ', '').strip()
        base_indent = len(lines[target_line]) - len(lines[target_line].lstrip())
        
        end_line = target_line + 1
        for i in range(target_line + 1, len(lines)):
            line = lines[i]
            if line.strip() and not line.startswith(' ' * (base_indent + 1)) and line.strip()[0] != '#':
                if line.strip()[0] not in ['@', '#']:  # Not a decorator or comment
                    if len(line) - len(line.lstrip()) <= base_indent:
                        end_line = i
                        break
            end_line = i + 1
        
        source = ''.join(lines[target_line:end_line])
        return (func_name, source, target_line + 1, end_line)
    except Exception as e:
        return None


def get_code_with_line_numbers(filepath, exclude_docstring=True, max_lines=None):
    """Get source code with line numbers for display.
    
    Returns list of tuples: (line_number, code_line)
    """
    try:
        all_code = get_source_code(filepath, exclude_docstring)
        lines = all_code.split('\n')
        
        if max_lines and len(lines) > max_lines:
            lines = lines[:max_lines]
        
        # Number the lines (accounting for skipped docstring lines)
        offset = 0
        if exclude_docstring:
            with open(filepath, 'r', encoding='utf-8') as f:
                original = f.read()
                for quote in ['"""', "'''"]:
                    if original.startswith(quote):
                        end_idx = original.find(quote, 3)
                        if end_idx != -1:
                            offset = original[:end_idx + 3].count('\n') + 1
                            break
        
        result = []
        for i, line in enumerate(lines, start=offset + 1):
            result.append((i, line))
        
        return result
    except Exception:
        return []

