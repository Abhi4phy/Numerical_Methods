"""
GUI Launcher — tkinter-based graphical interface.
===================================================
Dark-themed GUI with:
  • Three navigation modes (Browse / Learning Path / Search)
  • Rich-text theory panel with syntax highlighting
  • Rendered LaTeX equations via matplotlib
  • Runnable demos with captured output
"""

import os
import sys
import subprocess
import threading

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
    tk.Label(header, text="57 methods · 16 categories",
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

    nav_buttons = []
    for text, view in [("📂 Browse", "browse"),
                       ("📚 Learning Path", "learn"),
                       ("🔍 Search", "search")]:
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

    list_canvas = tk.Canvas(left_frame, bg=BG2, highlightthickness=0)
    list_scrollbar = ttk.Scrollbar(left_frame, orient="vertical",
                                    command=list_canvas.yview)
    list_inner = tk.Frame(list_canvas, bg=BG2)

    list_inner.bind("<Configure>",
                    lambda e: list_canvas.configure(
                        scrollregion=list_canvas.bbox("all")))
    list_canvas.create_window((0, 0), window=list_inner, anchor="nw")
    list_canvas.configure(yscrollcommand=list_scrollbar.set)

    list_scrollbar.pack(side="right", fill="y")
    list_canvas.pack(fill="both", expand=True)

    def _on_mousewheel(event):
        list_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    list_canvas.bind_all("<MouseWheel>", _on_mousewheel)

    # Right panel: details and output
    right_frame = tk.Frame(content, bg=BG)
    content.add(right_frame, minsize=400)

    detail_header = tk.Label(right_frame,
                             text="Select a method to view details",
                             font=FONT_B, bg=BG, fg=ACCENT3,
                             anchor="w", padx=10, pady=6)
    detail_header.pack(fill="x")

    # ── Description area ───────────────────────────────
    desc_frame = tk.Frame(right_frame, bg=BG)
    desc_frame.pack(fill="both", expand=True)

    desc_text = scrolledtext.ScrolledText(
        desc_frame, font=("Consolas", 10),
        bg=BG2, fg=FG, insertbackground=FG,
        wrap="word", relief="flat",
        padx=12, pady=10, spacing1=2, spacing3=2)
    desc_text.pack(fill="both", expand=True, padx=5, pady=2)

    # ── Rich-text tags ─────────────────────────────────
    desc_text.tag_configure("title",   font=FONT_LG,
                           foreground=ACCENT, spacing1=4, spacing3=2)
    desc_text.tag_configure("heading", font=("Consolas", 12, "bold"),
                           foreground=ACCENT2, spacing1=8, spacing3=2)
    desc_text.tag_configure("subheading", font=("Consolas", 10, "bold"),
                           foreground=ACCENT3, spacing1=4)
    desc_text.tag_configure("equation", font=("Consolas", 10),
                           foreground=PURPLE, background=BG3,
                           lmargin1=30, lmargin2=30,
                           spacing1=2, spacing3=2)
    desc_text.tag_configure("eq_header", font=FONT_EQ,
                           foreground=PURPLE, spacing1=6, spacing3=4)
    desc_text.tag_configure("eq_label", font=("Consolas", 10, "italic"),
                           foreground=PINK, lmargin1=20, spacing1=4)
    desc_text.tag_configure("bullet", font=("Consolas", 10),
                           foreground=FG, lmargin1=20, lmargin2=35)
    desc_text.tag_configure("start_section",
                           font=("Consolas", 12, "bold"),
                           foreground=ACCENT2, background=GREEN_BG,
                           spacing1=10, spacing3=4,
                           lmargin1=5, lmargin2=5)
    desc_text.tag_configure("separator", foreground=SEP_CLR)
    desc_text.tag_configure("normal", font=("Consolas", 10),
                           foreground=FG, lmargin1=10)
    desc_text.tag_configure("dim", font=FONT_SM,
                           foreground=FG_DIM, lmargin1=10)
    desc_text.tag_configure("code", font=FONT_SM,
                           foreground=ORANGE, background=BG3,
                           lmargin1=30, lmargin2=30)

    # Image references (prevent GC of PhotoImages)
    _eq_images = []

    # ── LaTeX renderer ─────────────────────────────────
    def _render_latex(latex_str, fontsize=14):
        """Render LaTeX → PNG → base64 → tk.PhotoImage."""
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            import io, base64

            fig = Figure(figsize=(LATEX_FIG_W, LATEX_FIG_H), dpi=LATEX_DPI)
            fig.patch.set_facecolor(BG2)
            fig.text(0.02, 0.5, f"${latex_str}$",
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
                    "  📐  Key Equations (Rendered)\n", "eq_header")
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

        desc_text.configure(state="disabled")

    # ── Welcome screen ─────────────────────────────────
    render_rich_text(
        "Numerical Methods for Physics\n"
        "==============================\n\n"
        "Welcome!  Browse categories on the left, or use the\n"
        "Learning Path for a recommended study order.\n\n"
        "Select any method to read its theory & equations, then\n"
        "click  \u25b6 Run Demo  to see it in action.\n\n"
        "Features\n"
        "--------\n"
        "- Rich formatted theory with highlighted equations\n"
        "- Rendered LaTeX equations (via matplotlib)\n"
        "- Runnable demos for every method\n"
        "- 7-stage learning roadmap\n"
        "- 16 categories \u00b7 58 implementations")

    # ── Button bar ─────────────────────────────────────
    btn_bar = tk.Frame(right_frame, bg=BG, pady=6)
    btn_bar.pack(fill="x", padx=5)

    selected_file = tk.StringVar(value="")

    def run_selected():
        fpath = selected_file.get()
        if not fpath or not os.path.exists(fpath):
            messagebox.showwarning("No file",
                                   "Please select a method first.")
            return

        desc_text.configure(state="normal")
        desc_text.delete("1.0", "end")
        desc_text.insert("1.0",
            f"Running {os.path.basename(fpath)}...\n\n")
        desc_text.configure(state="disabled")
        run_btn.configure(state="disabled")

        def _run():
            try:
                result = subprocess.run(
                    [sys.executable, fpath],
                    capture_output=True, text=True,
                    cwd=os.path.dirname(fpath),
                    timeout=SCRIPT_TIMEOUT
                )
                output = result.stdout
                if result.stderr:
                    output += "\n── stderr ──\n" + result.stderr
            except subprocess.TimeoutExpired:
                output = (f"⚠️ Script timed out after "
                          f"{SCRIPT_TIMEOUT} seconds.")
            except Exception as e:
                output = f"❌ Error: {e}"

            def _update():
                desc_text.configure(state="normal")
                desc_text.delete("1.0", "end")
                desc_text.insert("1.0",
                    f"── Output: {os.path.basename(fpath)} ──\n\n")
                desc_text.insert("end", output)
                desc_text.configure(state="disabled")
                run_btn.configure(state="normal")

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

    # ── View builders ──────────────────────────────────
    def clear_list():
        for w in list_inner.winfo_children():
            w.destroy()

    def make_file_button(parent, title, desc, filepath, indent=0):
        """Create a clickable file entry."""
        frame = tk.Frame(parent, bg=BG2, pady=1)
        frame.pack(fill="x", padx=(5 + indent*15, 5))

        exists = os.path.exists(filepath)
        mark = "✓" if exists else "✗"
        mark_color = ACCENT2 if exists else RED

        tk.Label(frame, text=mark, font=FONT_SM, bg=BG2,
                fg=mark_color, width=2).pack(side="left")

        btn = tk.Button(frame, text=title, font=FONT, bg=BG2, fg=FG,
                       activebackground=BG3, activeforeground=ACCENT,
                       relief="flat", anchor="w", cursor="hand2",
                       command=lambda: select_file(filepath, title, desc))
        btn.pack(side="left", fill="x", expand=True)

        tk.Label(frame, text=desc[:40], font=FONT_SM, bg=BG2,
                fg=FG_DIM, anchor="e").pack(side="right", padx=5)

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

    def show_search():
        clear_list()
        left_header.configure(text="  🔍 Search")

        search_frame = tk.Frame(list_inner, bg=BG2, pady=8)
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
