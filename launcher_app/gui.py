"""
GUI Launcher — tkinter-based graphical interface.
===================================================
Dark-themed GUI with:
    • Four navigation modes (Browse / Learning Path / Search / Discovery)
  • Rich-text theory panel with syntax highlighting
  • Rendered LaTeX equations via matplotlib
  • Runnable demos with captured output
"""

import os
import sys
import subprocess
import threading
import tempfile

from .config import (
    BASE_DIR, BG, BG2, BG3, FG, FG_DIM,
    ACCENT, ACCENT2, ACCENT3, RED,
    PURPLE, PINK, ORANGE, GREEN_BG, SEP_CLR,
    FONT, FONT_B, FONT_H, FONT_SM, FONT_LG, FONT_EQ,
    SCRIPT_TIMEOUT, LATEX_DPI, LATEX_FIG_W, LATEX_FIG_H,
)
from .catalog import CATEGORIES, LEARNING_PATH
from .equations import EQUATIONS, is_math_line
from .utils import get_file_docstring


def gui_launcher():
    """Launch the tkinter GUI."""
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Prevent popup windows

    root = tk.Tk()
    root.title("Numerical Methods for Physics")
    root.geometry("1050x700")
    root.configure(bg=BG)
    root.minsize(900, 600)

    # ── ttk Styles ─────────────────────────────────────
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("Cat.TButton", font=FONT, padding=6,
                    background=BG2, foreground=FG)
    style.map("Cat.TButton",
              background=[("active", BG3)],
              foreground=[("active", ACCENT)])
    style.configure("Run.TButton", font=FONT_B, padding=8,
                    background="#45475a", foreground=ACCENT2)
    style.map("Run.TButton",
              background=[("active", "#585b70")])
    style.configure("Nav.TButton", font=FONT, padding=4,
                    background=BG2, foreground=ACCENT)
    style.map("Nav.TButton",
              background=[("active", BG3)])

    # ── Header ─────────────────────────────────────────
    header = tk.Frame(root, bg=BG, pady=8)
    header.pack(fill="x")
    tk.Label(header, text="⚛  Numerical Methods for Physics",
             font=FONT_H, bg=BG, fg=ACCENT).pack(side="left", padx=15)
    tk.Label(header, text="58 methods · 16 categories",
             font=FONT_SM, bg=BG, fg=FG_DIM).pack(side="left", padx=10)

    # ── Navigation tabs ────────────────────────────────
    nav_frame = tk.Frame(root, bg=BG)
    nav_frame.pack(fill="x", padx=10)

    current_view = tk.StringVar(value="browse")

    def switch_view(view):
        current_view.set(view)
        for btn in nav_buttons:
            btn.configure(style="Nav.TButton")
        if view == "browse":
            nav_buttons[0].configure(style="Run.TButton")
            show_browse()
        elif view == "learn":
            nav_buttons[1].configure(style="Run.TButton")
            show_learning()
        elif view == "search":
            nav_buttons[2].configure(style="Run.TButton")
            show_search()
        elif view == "discover":
            nav_buttons[3].configure(style="Run.TButton")
            show_discovery()

    nav_buttons = []
    for text, view in [("📂 Browse", "browse"),
                       ("📚 Learning Path", "learn"),
                       ("🔍 Search", "search"),
                       ("🧭 Discovery", "discover")]:
        btn = ttk.Button(nav_frame, text=text, style="Nav.TButton",
                        command=lambda v=view: switch_view(v))
        btn.pack(side="left", padx=3, pady=4)
        nav_buttons.append(btn)

    # ── Main content area ──────────────────────────────
    content = tk.PanedWindow(root, orient="horizontal", bg=BG,
                             sashwidth=4, sashrelief="flat")
    content.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # Left panel: category / file list
    left_frame = tk.Frame(content, bg=BG2, width=360)
    content.add(left_frame, minsize=280)

    left_header = tk.Label(left_frame, text="Categories", font=FONT_B,
                           bg=BG2, fg=ACCENT, anchor="w", padx=10, pady=6)
    left_header.pack(fill="x")

    # Fixed (non-scrolling) controls area used by views like Search.
    search_top_frame = tk.Frame(left_frame, bg=BG2)

    # Create wrapper frame for canvas and scrollbar
    list_wrapper = tk.Frame(left_frame, bg=BG2)
    list_wrapper.pack(fill="both", expand=True)

    list_canvas = tk.Canvas(list_wrapper, bg=BG2, highlightthickness=0)
    list_scrollbar = ttk.Scrollbar(list_wrapper, orient="vertical",
                                    command=list_canvas.yview)
    list_inner = tk.Frame(list_canvas, bg=BG2)

    # Create window in canvas (this returns the window ID)
    canvas_window = list_canvas.create_window((0, 0), window=list_inner, anchor="nw", width=340)
    list_canvas.configure(yscrollcommand=list_scrollbar.set)

    # Grid layout for precise control
    list_canvas.grid(row=0, column=0, sticky="nsew")
    list_scrollbar.grid(row=0, column=1, sticky="ns")
    
    # Configure grid weights
    list_wrapper.grid_rowconfigure(0, weight=1)
    list_wrapper.grid_columnconfigure(0, weight=1)

    def _update_scroll_region():
        """Recalculate scroll region when content changes."""
        list_canvas.configure(scrollregion=list_canvas.bbox("all"))
        # Make inner frame fill canvas width
        cwidth = list_canvas.winfo_width()
        if cwidth > 1:
            list_canvas.itemconfigure(canvas_window, width=cwidth - 4)

    # Bindings for updating scroll region
    list_inner.bind("<Configure>", lambda e: _update_scroll_region())
    list_canvas.bind("<Configure>", lambda e: _update_scroll_region())

    def _on_list_mousewheel(event):
        """Scroll method list only (don't sync with theory/code)."""
        list_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        return "break"  # Stop event propagation
    
    def _bind_mousewheel_recursive(widget):
        """Recursively bind mousewheel to widget and all children."""
        widget.bind("<MouseWheel>", _on_list_mousewheel)
        for child in widget.winfo_children():
            _bind_mousewheel_recursive(child)
    
    # Bind mousewheel to list canvas and all descendants
    list_canvas.bind("<MouseWheel>", _on_list_mousewheel)
    list_inner.bind("<MouseWheel>", _on_list_mousewheel)
    _bind_mousewheel_recursive(list_inner)

    # Right panel: details and output
    right_frame = tk.Frame(content, bg=BG)
    content.add(right_frame, minsize=400)

    detail_header = tk.Label(right_frame,
                             text="Select a method to view details",
                             font=FONT_B, bg=BG, fg=ACCENT3,
                             anchor="w", padx=10, pady=6)
    detail_header.pack(fill="x")

    # ── Split main content area (Theory | Code) ────────
    main_content = tk.PanedWindow(right_frame, orient="horizontal", 
                                  bg=BG, sashwidth=4, sashrelief="flat")
    main_content.pack(fill="both", expand=True, padx=5, pady=(0, 5))

    # ── LEFT: Theory panel ────────────────────────────
    theory_frame = tk.Frame(main_content, bg=BG)
    main_content.add(theory_frame, minsize=250)

    theory_label = tk.Label(theory_frame, text="Theory", 
                            font=FONT_B, bg=BG, fg=ACCENT2,
                            anchor="w", padx=5, pady=3)
    theory_label.pack(fill="x")

    desc_text = scrolledtext.ScrolledText(
        theory_frame, font=("Consolas", 9),
        bg=BG2, fg=FG, insertbackground=FG,
        wrap="word", relief="flat",
        padx=8, pady=8, spacing1=1, spacing3=1)
    desc_text.pack(fill="both", expand=True)

    # ── RIGHT: Code panel ────────────────────────────
    code_frame = tk.Frame(main_content, bg=BG)
    main_content.add(code_frame, minsize=150)

    code_label = tk.Label(code_frame, text="Implementation", 
                          font=FONT_B, bg=BG, fg=ACCENT,
                          anchor="w", padx=5, pady=3)
    code_label.pack(fill="x")

    code_text = scrolledtext.ScrolledText(
        code_frame, font=("Consolas", 8),
        bg="#0f0f1e", fg="#d0d0e0", insertbackground=FG,
        wrap="none", relief="flat",
        padx=8, pady=8, spacing1=0, spacing3=0)
    code_text.pack(fill="both", expand=True)

    # ── Synced scrolling between theory and code panels ────
    # ── Independent scrolling for theory and code panels ────
    def _on_theory_mousewheel(event):
        """Scroll theory panel independently."""
        desc_text.yview_scroll(int(-1*(event.delta/120)), "units")
        return "break"

    def _on_code_mousewheel(event):
        """Scroll code panel independently."""
        code_text.yview_scroll(int(-1*(event.delta/120)), "units")
        return "break"

    # Bind mousewheel to both panels for independent scrolling
    desc_text.bind("<MouseWheel>", _on_theory_mousewheel)
    code_text.bind("<MouseWheel>", _on_code_mousewheel)

    # ── Rich-text tags ─────────────────────────────────
    desc_text.tag_configure("title",   font=FONT_LG,
                           foreground=ACCENT, spacing1=4, spacing3=2)
    desc_text.tag_configure("heading", font=("Consolas", 11, "bold"),
                           foreground=ACCENT2, spacing1=8, spacing3=2)
    desc_text.tag_configure("subheading", font=("Consolas", 9, "bold"),
                           foreground=ACCENT3, spacing1=4)
    desc_text.tag_configure("equation", font=("Consolas", 9),
                           foreground=PURPLE, background=BG3,
                           lmargin1=20, lmargin2=20,
                           spacing1=2, spacing3=2)
    desc_text.tag_configure("eq_header", font=("Consolas", 11, "bold"),
                           foreground=PURPLE, spacing1=6, spacing3=4)
    desc_text.tag_configure("eq_label", font=("Consolas", 9, "italic"),
                           foreground=PINK, lmargin1=15, spacing1=4)
    desc_text.tag_configure("bullet", font=("Consolas", 9),
                           foreground=FG, lmargin1=15, lmargin2=25)
    desc_text.tag_configure("start_section",
                           font=("Consolas", 10, "bold"),
                           foreground=ACCENT2, background=GREEN_BG,
                           spacing1=6, spacing3=3,
                           lmargin1=5, lmargin2=5)
    desc_text.tag_configure("separator", foreground=SEP_CLR)
    desc_text.tag_configure("normal", font=("Consolas", 9),
                           foreground=FG, lmargin1=8)
    desc_text.tag_configure("dim", font=("Consolas", 8),
                           foreground=FG_DIM, lmargin1=8)
    desc_text.tag_configure("code", font=("Consolas", 8),
                           foreground=ORANGE, background=BG3,
                           lmargin1=15, lmargin2=15)

    # ── Metadata section tags ──────────────────────────
    desc_text.tag_configure("prereq_header", 
                           font=("Consolas", 10, "bold"),
                           foreground="#a6d189", spacing1=6, spacing3=2)
    desc_text.tag_configure("app_header",
                           font=("Consolas", 10, "bold"),
                           foreground="#89dceb", spacing1=6, spacing3=2)
    desc_text.tag_configure("pitfall_header",
                           font=("Consolas", 10, "bold"),
                           foreground="#f38ba8", spacing1=6, spacing3=2)
    desc_text.tag_configure("complexity_header",
                           font=("Consolas", 10, "bold"),
                           foreground="#f9e2af", spacing1=6, spacing3=2)
    desc_text.tag_configure("related_header",
                           font=("Consolas", 10, "bold"),
                           foreground="#cba6f7", spacing1=6, spacing3=2)
    desc_text.tag_configure("metadata_item",
                           font=("Consolas", 9),
                           foreground=FG, lmargin1=20, lmargin2=35)

    # ── Code syntax highlighting tags (Pygments palette) ────
    code_text.tag_configure("keyword",    foreground="#ff79c6")  # Pink keywords
    code_text.tag_configure("string",     foreground="#f1fa8c")  # Green strings
    code_text.tag_configure("comment",    foreground="#6272a4")  # Dark blue comments
    code_text.tag_configure("function",   foreground="#50fa7b")  # Light green functions
    code_text.tag_configure("number",     foreground="#bd93f9")  # Purple numbers
    code_text.tag_configure("operator",   foreground="#ff79c6")  # Pink operators
    code_text.tag_configure("builtin",    foreground="#8be9fd")  # Cyan builtins
    code_text.tag_configure("class",      foreground="#50fa7b")  # Green classes
    code_text.tag_configure("lineno",     foreground="#44475a")  # Gray line numbers
    code_text.tag_configure("normal",     foreground="#f8f8f2")  # Default

    _code_syntax_tags = (
        "keyword", "string", "comment", "function",
        "number", "operator", "builtin", "class"
    )

    # Image references (prevent GC of PhotoImages)
    _eq_images = []
    _highlight_job = [None]

    def _sanitize_latex_for_mathtext(latex_str):
        """Convert common LaTeX commands unsupported by matplotlib mathtext."""
        s = latex_str.strip()
        if s.startswith("$") and s.endswith("$"):
            s = s[1:-1]

        replacements = {
            r"\tfrac": r"\frac",
            r"\dfrac": r"\frac",
            r"\bigl": r"\left",
            r"\bigr": r"\right",
            r"\Bigl": r"\left",
            r"\Bigr": r"\right",
            r"\biggl": r"\left",
            r"\biggr": r"\right",
            r"\text": r"\mathrm",
        }
        for old, new in replacements.items():
            s = s.replace(old, new)
        return s

    # ── LaTeX renderer ─────────────────────────────────
    def _render_latex(latex_str, fontsize=14):
        """Render LaTeX → PNG → base64 → tk.PhotoImage."""
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            import io, base64

            clean_latex = _sanitize_latex_for_mathtext(latex_str)

            fig = Figure(figsize=(LATEX_FIG_W, LATEX_FIG_H), dpi=LATEX_DPI)
            fig.patch.set_facecolor(BG2)
            fig.text(0.02, 0.5, f"${clean_latex}$",
                     fontsize=fontsize, color=PURPLE,
                     va='center', ha='left')

            canvas = FigureCanvasAgg(fig)
            canvas.draw()

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=LATEX_DPI,
                        bbox_inches='tight', pad_inches=0.06,
                        facecolor=BG2, edgecolor='none')
            buf.seek(0)

            img_data = base64.b64encode(buf.read()).decode('ascii')
            return tk.PhotoImage(data=img_data)
        except Exception:
            return None

    # ── Code highlighting with Pygments ────────────────
    def _highlight_code(source_code):
        """Apply syntax highlighting to code using Pygments."""
        try:
            from pygments.lexers import PythonLexer
            from pygments.token import Token
            
            # Use Pygments tokenizer
            lexer = PythonLexer()
            tokens = list(lexer.get_tokens(source_code))
            
            # Map token types to our tags
            token_map = {
                Token.Keyword:       "keyword",
                Token.String:        "string",
                Token.Comment:       "comment",
                Token.Name.Function: "function",
                Token.Number:        "number",
                Token.Operator:      "operator",
                Token.Name.Builtin:  "builtin",
                Token.Name.Class:    "class",
            }
            
            result = []
            for token_type, value in tokens:
                # Get the most specific token category
                tag = "normal"
                for tk_type, tk_tag in token_map.items():
                    if token_type in tk_type:
                        tag = tk_tag
                        break
                result.append((value, tag))
            
            return result
        except Exception:
            # Fallback: plain text
            return [(source_code, "normal")]

    def _apply_syntax_highlighting(source_code):
        """Apply token tags to the code editor buffer."""
        for tag in _code_syntax_tags:
            code_text.tag_remove(tag, "1.0", "end")

        offset = 0
        for token_text, tag in _highlight_code(source_code):
            length = len(token_text)
            if length <= 0:
                continue
            if tag in _code_syntax_tags:
                start_idx = f"1.0 + {offset} chars"
                end_idx = f"1.0 + {offset + length} chars"
                code_text.tag_add(tag, start_idx, end_idx)
            offset += length

    def _schedule_highlight(_event=None):
        """Debounce highlighting while typing to keep the UI responsive."""
        if _highlight_job[0] is not None:
            try:
                root.after_cancel(_highlight_job[0])
            except Exception:
                pass
        _highlight_job[0] = root.after(120, _recolor_code_editor)

    def _recolor_code_editor():
        """Recompute and apply syntax highlighting for current code text."""
        _highlight_job[0] = None
        source_code = code_text.get("1.0", "end-1c")
        if source_code:
            _apply_syntax_highlighting(source_code)

    def _display_code(filepath):
        """Display source code in code_text with syntax highlighting."""
        code_text.configure(state="normal")
        code_text.delete("1.0", "end")
        
        try:
            from .utils import get_code_with_line_numbers
            
            # Get source with line numbers (no limit to avoid truncating functions)
            lines = get_code_with_line_numbers(filepath, exclude_docstring=True, max_lines=None)
            
            if not lines:
                code_text.insert("1.0", "# Could not read code\n", "comment")
                code_text.configure(state="normal")
                return
            
            # Store original code without line numbers for reset functionality
            source_code = '\n'.join(line[1] for line in lines)
            original_code[0] = source_code

            # Display code without line numbers, then colorize tokens.
            code_text.insert("1.0", source_code)
            _apply_syntax_highlighting(source_code)
            
            # Keep code editable for experimentation
            code_text.configure(state="normal")
        except Exception as e:
            code_text.insert("end", f"# Error loading code: {e}\n", "comment")
            code_text.configure(state="normal")

    # Re-apply highlighting as users edit code in the panel.
    code_text.bind("<KeyRelease>", _schedule_highlight, add=True)

    # ── Rich-text rendering engine ─────────────────────
    def render_rich_text(text_content, filepath=None):
        """Parse docstring and insert with rich formatting + LaTeX."""
        _eq_images.clear()
        desc_text.configure(state="normal")
        desc_text.delete("1.0", "end")

        lines = text_content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""

            # Title (followed by ===)
            if nxt and len(nxt) >= 3 and all(c == '=' for c in nxt):
                desc_text.insert("end", stripped + "\n", "title")
                desc_text.insert("end",
                    "═" * min(len(stripped), 55) + "\n\n", "separator")
                i += 2; continue

            # Heading (followed by --- or ━━━)
            if nxt and len(nxt) >= 3 and all(c in '-━' for c in nxt):
                desc_text.insert("end", "\n" + stripped + "\n", "heading")
                desc_text.insert("end",
                    "─" * min(len(stripped), 50) + "\n", "separator")
                i += 2; continue

            # "Where to start" section
            if stripped.lower().startswith('where to start'):
                desc_text.insert("end", "\n ▶ " + stripped + "\n",
                                 "start_section")
                i += 1; continue

            # Bold markers  **text** / **text:**
            if stripped.startswith("**"):
                end_b = stripped.find("**", 2)
                if end_b > 0:
                    bold = stripped[2:end_b]
                    rest = stripped[end_b + 2:]
                    desc_text.insert("end", "  " + bold, "subheading")
                    if rest:
                        desc_text.insert("end", rest, "normal")
                    desc_text.insert("end", "\n")
                    i += 1; continue

            # Equation lines (Unicode math symbols)
            if is_math_line(stripped):
                desc_text.insert("end", "  " + stripped + "\n", "equation")
                i += 1; continue

            # Bullet points
            if stripped.startswith("- ") or stripped.startswith("* "):
                desc_text.insert("end", "  • " + stripped[2:] + "\n",
                                 "bullet")
                i += 1; continue

            # Numbered items (1. / 1) / 2. etc.)
            if stripped and stripped[0].isdigit() and len(stripped) > 2:
                if stripped[1] in '.)' and stripped[2] == ' ':
                    desc_text.insert("end", "  " + stripped + "\n",
                                     "bullet")
                    i += 1; continue
                if (stripped[1].isdigit() and len(stripped) > 3
                        and stripped[2] in '.)'):
                    desc_text.insert("end", "  " + stripped + "\n",
                                     "bullet")
                    i += 1; continue

            # Separator lines
            if (stripped and len(stripped) >= 3
                    and all(c in '═─━-=~' for c in stripped)):
                desc_text.insert("end", "─" * 40 + "\n", "separator")
                i += 1; continue

            # Empty line
            if not stripped:
                desc_text.insert("end", "\n")
                i += 1; continue

            # Normal text
            desc_text.insert("end", "  " + stripped + "\n", "normal")
            i += 1

        # ── Rendered LaTeX equations (if available) ────
        if filepath:
            fname = os.path.basename(filepath)
            if fname in EQUATIONS:
                desc_text.insert("end", "\n\n")
                desc_text.insert("end", "━" * 45 + "\n", "separator")
                desc_text.insert("end",
                    "  📐  Key Equations\n", "eq_header")
                desc_text.insert("end", "━" * 45 + "\n\n", "separator")

                for label, latex in EQUATIONS[fname]:
                    if label:
                        desc_text.insert("end",
                            f"  {label}:\n", "eq_label")
                    photo = _render_latex(latex)
                    if photo:
                        _eq_images.append(photo)
                        desc_text.insert("end", "     ")
                        desc_text.image_create("end", image=photo)
                        desc_text.insert("end", "\n\n")
                    else:
                        desc_text.insert("end",
                            f"      {latex}\n\n", "equation")

            # ── Metadata sections (from METHOD_METADATA) ────
            from .equations import METHOD_METADATA
            if fname in METHOD_METADATA:
                meta = METHOD_METADATA[fname]
                
                # Prerequisites
                if meta.get("prerequisites"):
                    desc_text.insert("end", "\n")
                    desc_text.insert("end", "━" * 35 + "\n", "separator")
                    desc_text.insert("end", "  📋  Prerequisites\n", "prereq_header")
                    for prereq in meta["prerequisites"]:
                        prereq = prereq.strip()
                        if prereq:
                            desc_text.insert("end", "  • " + prereq + "\n", "metadata_item")
                    desc_text.insert("end", "\n")
                
                # Complexity
                if meta.get("complexity"):
                    desc_text.insert("end", "⏱️   Complexity\n", "complexity_header")
                    text_val = meta["complexity"]
                    if isinstance(text_val, list):
                        for line in text_val:
                            line = line.strip()
                            if line:
                                desc_text.insert("end", "  " + line + "\n", "metadata_item")
                    else:
                        desc_text.insert("end", "  " + str(text_val) + "\n", "metadata_item")
                    desc_text.insert("end", "\n")
                
                # Applications
                if meta.get("applications"):
                    desc_text.insert("end", "🌍  Applications\n", "app_header")
                    for app in meta["applications"][:2]:  # First 2 only to save space
                        app = app.strip()
                        if app:
                            desc_text.insert("end", "  • " + app + "\n", "metadata_item")
                    desc_text.insert("end", "\n")
                
                # Pitfalls
                if meta.get("pitfalls"):
                    desc_text.insert("end", "⚠️   Pitfalls\n", "pitfall_header")
                    for pitfall in meta["pitfalls"][:1]:  # First only to save space
                        pitfall = pitfall.strip()
                        if pitfall:
                            desc_text.insert("end", "  ✗ " + pitfall + "\n", "metadata_item")
                    desc_text.insert("end", "\n")

        desc_text.configure(state="disabled")

        # Display code if filepath provided
        if filepath and os.path.exists(filepath):
            _display_code(filepath)


    # ── Welcome screen ─────────────────────────────────
    render_rich_text(
        "Numerical Methods for Physics\n"
        "==============================\n\n"
        "Welcome! Browse categories on the left\n"
        "or use Learning Path for guided order.\n\n"
        "Select any method to view theory\n"
        "alongside its implementation.\n\n"
        "Features\n"
        "--------\n"
        "✓ Side-by-side theory + code\n"
        "✓ Rendered LaTeX equations\n"
        "✓ Syntax-highlighted source\n"
        "✓ 7-stage learning roadmap\n"
        "✓ 16 categories · 58 methods")

    # ── Button bar ─────────────────────────────────────
    btn_bar = tk.Frame(right_frame, bg=BG, pady=6, relief="solid", bd=1)
    btn_bar.pack(fill="x", padx=5)

    selected_file = tk.StringVar(value="")
    original_code = [""]  # Store original code as list for mutability

    def run_selected():
        fpath = selected_file.get()
        if not fpath:
            messagebox.showwarning("No file",
                                   "Please select a method first.")
            return

        desc_text.configure(state="normal")
        desc_text.delete("1.0", "end")
        desc_text.insert("1.0",
            f"Running {os.path.basename(fpath)}...\n\n")
        desc_text.configure(state="disabled")

        def _run():
            output = ""  # Initialize output to ensure it's always defined
            try:
                # Get edited code from code_text widget
                edited_code = code_text.get("1.0", "end-1c")
                
                if not edited_code.strip():
                    output = "❌ Error: No code to run"
                else:
                    # Get the working directory for proper imports
                    work_dir = os.path.dirname(selected_file.get()) or BASE_DIR
                    
                    # Prepend sys.path setup so imports work from the original file's directory
                    code_with_imports = f"import sys\nsys.path.insert(0, {repr(work_dir)})\n\n{edited_code}"
                    
                    # Create temporary file with edited code
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', 
                                                     delete=False, encoding='utf-8') as f:
                        f.write(code_with_imports)
                        temp_file = f.name
                    
                    try:
                        # Set UTF-8 encoding for subprocess to handle Unicode characters
                        env = os.environ.copy()
                        env['PYTHONIOENCODING'] = 'utf-8'
                        
                        result = subprocess.run(
                            [sys.executable, temp_file],
                            capture_output=True,
                            cwd=work_dir,
                            timeout=SCRIPT_TIMEOUT,
                            env=env,
                            encoding='utf-8',
                            errors='replace'
                        )
                        output = result.stdout if result.stdout else "(No output)"
                        if result.stderr:
                            output += "\n── stderr ──\n" + result.stderr
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
                        
            except subprocess.TimeoutExpired:
                output = (f"⚠️ Script timed out after "
                          f"{SCRIPT_TIMEOUT} seconds.")
            except Exception as e:
                output = f"❌ Error: {e}"

            def _update():
                desc_text.configure(state="normal")
                desc_text.delete("1.0", "end")
                desc_text.insert("1.0",
                    f"── Output: {os.path.basename(selected_file.get())} ──\n\n")
                desc_text.insert("end", output)
                desc_text.configure(state="disabled")

            root.after(0, _update)

        threading.Thread(target=_run, daemon=True).start()

    def view_docstring():
        fpath = selected_file.get()
        if not fpath or not os.path.exists(fpath):
            return
        docstring = get_file_docstring(fpath)
        render_rich_text(docstring, fpath)

    def show_equations():
        """Show only the rendered LaTeX equations for selected method."""
        fpath = selected_file.get()
        if not fpath or not os.path.exists(fpath):
            messagebox.showinfo("No file",
                                "Please select a method first.")
            return
        fname = os.path.basename(fpath)
        if fname not in EQUATIONS:
            desc_text.configure(state="normal")
            desc_text.delete("1.0", "end")
            _eq_images.clear()
            desc_text.insert("end",
                "\n  No rendered equations available for this method."
                "\n\n", "normal")
            desc_text.insert("end",
                "  Click '📄 View Theory' to see the full description"
                "\n", "dim")
            desc_text.insert("end",
                "  with inline equations from the source code.\n",
                "dim")
            desc_text.configure(state="disabled")
            return
        _eq_images.clear()
        desc_text.configure(state="normal")
        desc_text.delete("1.0", "end")
        title = fname.replace('.py', '').replace('_', ' ').title()
        desc_text.insert("end",
            f"\n  📐  {title} \u2014 Key Equations\n", "eq_header")
        desc_text.insert("end", "━" * 50 + "\n\n", "separator")
        for label, latex in EQUATIONS[fname]:
            if label:
                desc_text.insert("end", f"  {label}\n", "eq_label")
            photo = _render_latex(latex, fontsize=16)
            if photo:
                _eq_images.append(photo)
                desc_text.insert("end", "     ")
                desc_text.image_create("end", image=photo)
                desc_text.insert("end", "\n\n")
            else:
                desc_text.insert("end",
                    f"    {latex}\n\n", "equation")
        desc_text.configure(state="disabled")

    run_btn = ttk.Button(btn_bar, text="▶  Run Demo",
                        style="Run.TButton", command=run_selected)
    run_btn.pack(side="left", padx=4)

    view_btn = ttk.Button(btn_bar, text="📄 View Theory",
                         style="Nav.TButton", command=view_docstring)
    view_btn.pack(side="left", padx=4)

    eq_btn = ttk.Button(btn_bar, text="📐 Equations",
                        style="Nav.TButton", command=show_equations)
    eq_btn.pack(side="left", padx=4)

    def reset_code():
        """Reset code to original version."""
        if not original_code[0]:
            messagebox.showinfo("No code", "No modification to reset.")
            return
        code_text.configure(state="normal")
        code_text.delete("1.0", "end")
        code_text.insert("1.0", original_code[0])
        code_text.configure(state="normal")
        messagebox.showinfo("Reset", "Code reset to original version.")

    reset_btn = ttk.Button(btn_bar, text="🔄 Reset Code",
                          style="Nav.TButton", command=reset_code)
    reset_btn.pack(side="left", padx=4)

    # ── View builders ──────────────────────────────────
    def clear_list():
        for w in search_top_frame.winfo_children():
            w.destroy()
        search_top_frame.pack_forget()
        for w in list_inner.winfo_children():
            w.destroy()

    def _refresh_list_bindings():
        """Re-bind mousewheel to all items after list is rebuilt."""
        _bind_mousewheel_recursive(list_inner)

    def make_file_button(parent, title, desc, filepath, indent=0, on_click=None):
        """Create a clickable file entry."""
        frame = tk.Frame(parent, bg=BG2, pady=1)
        frame.pack(fill="x", padx=(5 + indent*15, 5))

        exists = os.path.exists(filepath)
        mark = "✓" if exists else "✗"
        mark_color = ACCENT2 if exists else RED

        mark_lbl = tk.Label(frame, text=mark, font=FONT_SM, bg=BG2,
                fg=mark_color, width=2)
        mark_lbl.pack(side="left")

        # Use custom on_click handler if provided, otherwise use select_file
        if on_click is None:
            on_click = lambda: select_file(filepath, title, desc)

        btn = tk.Button(frame, text=title, font=FONT, bg=BG2, fg=FG,
                       activebackground=BG3, activeforeground=ACCENT,
                       relief="flat", anchor="w", cursor="hand2",
                       command=on_click)
        btn.pack(side="left", fill="x", expand=True)

        desc_lbl = tk.Label(frame, text=desc[:40], font=FONT_SM, bg=BG2,
                fg=FG_DIM, anchor="e")
        desc_lbl.pack(side="right", padx=5)

        # Bind mousewheel to all widgets in this button row
        for widget in [frame, mark_lbl, btn, desc_lbl]:
            widget.bind("<MouseWheel>", _on_list_mousewheel)

    def select_file(filepath, title, desc):
        selected_file.set(filepath)
        detail_header.configure(text=f"  {title}")

        if os.path.exists(filepath):
            docstring = get_file_docstring(filepath)
            render_rich_text(docstring, filepath)
        else:
            desc_text.configure(state="normal")
            desc_text.delete("1.0", "end")
            desc_text.insert("1.0", "  File not found.", "normal")
            desc_text.configure(state="disabled")

    def show_browse():
        clear_list()
        left_header.configure(text="  Categories")

        for cat in CATEGORIES:
            cat_frame = tk.Frame(list_inner, bg=BG3, pady=4)
            cat_frame.pack(fill="x", padx=3, pady=(6, 1))

            tk.Label(cat_frame,
                    text=f"  {cat['icon']}  {cat['name']}",
                    font=FONT_B, bg=BG3, fg=ACCENT,
                    anchor="w").pack(fill="x", padx=5)

            for fname, title, desc in cat["files"]:
                filepath = os.path.join(BASE_DIR, cat["folder"], fname)
                make_file_button(list_inner, title, desc,
                                filepath, indent=1)

        _refresh_list_bindings()

    def show_learning():
        clear_list()
        left_header.configure(text="  📚 Learning Path")

        step = 1
        for stage_name, files in LEARNING_PATH:
            stage_frame = tk.Frame(list_inner, bg=BG3, pady=4)
            stage_frame.pack(fill="x", padx=3, pady=(8, 1))
            tk.Label(stage_frame, text=f"  {stage_name}",
                    font=FONT_B, bg=BG3, fg=ACCENT3,
                    anchor="w").pack(fill="x", padx=5)

            for folder, fname, desc in files:
                filepath = os.path.join(BASE_DIR, folder, fname)
                title = (f"{step}. "
                         f"{fname.replace('.py','').replace('_',' ').title()}")
                make_file_button(list_inner, title, desc,
                                filepath, indent=1)
                step += 1

        _refresh_list_bindings()

    def show_search():
        clear_list()
        left_header.configure(text="  🔍 Search")

        search_top_frame.pack(fill="x", padx=5, pady=(4, 2))
        search_frame = tk.Frame(search_top_frame, bg=BG2, pady=4)
        search_frame.pack(fill="x", padx=5)

        tk.Label(search_frame, text="Search:", font=FONT,
                bg=BG2, fg=FG).pack(side="left", padx=5)

        search_var = tk.StringVar()
        search_entry = tk.Entry(search_frame, textvariable=search_var,
                               font=FONT, bg=BG3, fg=FG,
                               insertbackground=FG, relief="flat")
        search_entry.pack(side="left", fill="x", expand=True, padx=5)
        search_entry.focus_set()

        results_frame = tk.Frame(list_inner, bg=BG2)
        results_frame.pack(fill="x")

        def do_search(*_):
            for w in results_frame.winfo_children():
                w.destroy()

            query = search_var.get().strip().lower()
            if len(query) < 2:
                return

            count = 0
            for cat in CATEGORIES:
                for fname, title, desc in cat["files"]:
                    searchable = (f"{title} {desc} {fname} "
                                  f"{cat['name']}").lower()
                    if query in searchable:
                        filepath = os.path.join(
                            BASE_DIR, cat["folder"], fname)
                        make_file_button(results_frame,
                            f"[{cat['id']}] {title}", desc, filepath)
                        count += 1

            if count == 0:
                for cat in CATEGORIES:
                    for fname, title, desc in cat["files"]:
                        filepath = os.path.join(
                            BASE_DIR, cat["folder"], fname)
                        if os.path.exists(filepath):
                            docstring = get_file_docstring(
                                filepath).lower()
                            if query in docstring:
                                make_file_button(results_frame,
                                    f"[{cat['id']}] {title}",
                                    desc, filepath)
                                count += 1

            if count == 0:
                tk.Label(results_frame,
                        text=f"  No results for '{query}'",
                        font=FONT, bg=BG2,
                        fg=FG_DIM).pack(padx=10, pady=10)

        search_var.trace_add("write", do_search)
        search_entry.bind("<Return>", do_search)

        _refresh_list_bindings()

    # ── Discovery & Recommendations Panel ──────────────
    def show_discovery():
        """Display discovery dashboard with recommendations."""
        clear_list()
        left_header.configure(text="  🧭 Discovery & Advisor")
        
        from .discovery_engine import (
            LearningAdvisor, get_recommendations, search_methods,
            get_method_info
        )
        
        advisor = LearningAdvisor()
        
        # Title
        tk.Label(list_inner, text="Discovery Dashboard",
                font=FONT_LG, bg=BG2, fg=ACCENT,
                anchor="w", padx=10, pady=10).pack(fill="x")
        
        # Section 1: Current method insights
        fpath = selected_file.get()
        
        if fpath and os.path.exists(fpath):
            fname = os.path.basename(fpath)
            method_info = get_method_info(fname)
            
            # Current method section
            frame1 = tk.Frame(list_inner, bg=BG3, padx=10, pady=8)
            frame1.pack(fill="x", padx=5, pady=(5, 10))
            
            tk.Label(frame1, text="📖 You're Learning:",
                    font=FONT_B, bg=BG3, fg=ACCENT2).pack(anchor="w")
            tk.Label(frame1, text=method_info["title"],
                    font=FONT_B, bg=BG3, fg=ACCENT3).pack(anchor="w", padx=15)
            tk.Label(frame1, text=f"Category: {method_info['category']}",
                    font=FONT_SM, bg=BG3, fg=FG_DIM).pack(anchor="w", padx=15)
            tk.Label(frame1, text=f"Difficulty: {method_info['difficulty'].title()}",
                    font=FONT_SM, bg=BG3, fg=FG).pack(anchor="w", padx=15)
            
            # Section 2: Recommendations
            recs = get_recommendations(fname)
            
            if recs["prerequisites"]:
                frame_pre = tk.Frame(list_inner, bg=BG2, padx=10, pady=8)
                frame_pre.pack(fill="x", padx=5, pady=(0, 5))
                tk.Label(frame_pre, text="📚 Prerequisites First:",
                        font=FONT_B, bg=BG2, fg=ORANGE).pack(anchor="w")
                for fname_pre, title_pre, reason_pre in recs["prerequisites"][:2]:
                    tk.Label(frame_pre, text=f"  • {title_pre}",
                            font=FONT_SM, bg=BG2, fg=FG).pack(anchor="w", padx=15)
            
            if recs["related"]:
                frame_rel = tk.Frame(list_inner, bg=BG2, padx=10, pady=8)
                frame_rel.pack(fill="x", padx=5, pady=(0, 5))
                tk.Label(frame_rel, text="🔗 Similar Methods:",
                        font=FONT_B, bg=BG2, fg=ACCENT2).pack(anchor="w")
                for fname_rel, title_rel, reason_rel in recs["related"][:2]:
                    tk.Label(frame_rel, text=f"  • {title_rel}",
                            font=FONT_SM, bg=BG2, fg=FG).pack(anchor="w", padx=15)
            
            if recs["next_steps"]:
                frame_next = tk.Frame(list_inner, bg=BG2, padx=10, pady=8)
                frame_next.pack(fill="x", padx=5, pady=(0, 5))
                tk.Label(frame_next, text="⏭️ Next Steps:",
                        font=FONT_B, bg=BG2, fg=GREEN_BG).pack(anchor="w")
                for fname_next, title_next, reason_next in recs["next_steps"][:2]:
                    tk.Label(frame_next, text=f"  • {title_next}",
                            font=FONT_SM, bg=BG2, fg=FG).pack(anchor="w", padx=15)
        else:
            # No method selected - show search
            tk.Label(list_inner, 
                    text="Select a method to see recommendations",
                    font=FONT_SM, bg=BG2, fg=FG_DIM,
                    anchor="w", padx=10, pady=20).pack(fill="x")
        
        # Learning advisor stats
        frame_stats = tk.Frame(list_inner, bg=BG3, padx=10, pady=8)
        frame_stats.pack(fill="x", padx=5, pady=(10, 0))
        tk.Label(frame_stats, text="📊 Learning Stats:",
                font=FONT_B, bg=BG3, fg=ACCENT3).pack(anchor="w")
        
        dist = advisor.get_difficulty_distribution()
        stats_text = (
            f"  Beginner: {dist['beginner']} methods\n"
            f"  Intermediate: {dist['intermediate']} methods\n"
            f"  Advanced: {dist['advanced']} methods"
        )
        tk.Label(frame_stats, text=stats_text,
                font=FONT_SM, bg=BG3, fg=FG, justify="left").pack(anchor="w", padx=15, pady=(5, 10))

        _refresh_list_bindings()

    # ── Initialize ─────────────────────────────────────
    switch_view("browse")

    # ── Status bar ─────────────────────────────────────
    status = tk.Frame(root, bg=BG3, height=24)
    status.pack(fill="x", side="bottom")
    tk.Label(status,
             text=("  Numerical Methods for Physics · "
                   "Python + NumPy · 2024-2026"),
             font=FONT_SM, bg=BG3, fg=FG_DIM,
             anchor="w").pack(fill="x", padx=5)

    root.mainloop()
