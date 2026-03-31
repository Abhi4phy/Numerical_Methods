# Architecture — Numerical Methods Launcher

## Overview

The launcher provides dual-mode access (CLI + GUI) to **58 numerical method
implementations** across **16 categories**, with a **7-stage learning path**,
**rich-text theory display**, and **rendered LaTeX equations**.

---

## Package Structure

```
Numerical_Methods/
│
├── launcher.py                 ← Backward-compatible entry point (thin wrapper)
│
├── launcher_app/               ← Application package (modular architecture)
│   ├── __init__.py             Package metadata & version
│   ├── __main__.py             `python -m launcher_app` entry
│   ├── main.py                 Mode detection (CLI / GUI / auto)
│   ├── config.py               Paths, colours, fonts, runtime limits
│   ├── catalog.py              CATEGORIES (16), LEARNING_PATH (7 stages)
│   ├── equations.py            EQUATIONS database (58 methods, ~98 formulas)
│   ├── utils.py                File I/O, subprocess execution
│   ├── cli.py                  Terminal-based interactive menus
│   └── gui.py                  Tkinter GUI with rich-text rendering
│
├── 01_Linear_Algebra/          10 implementations
├── 02_Differential_Equations/   9 implementations
├── 03_Numerical_Integration/    4 implementations
├── 04_Interpolation_Approxim…/  5 implementations
├── 05_Root_Finding/             4 implementations
├── 06_Optimization/             4 implementations
├── 07_Numerical_Linear_Systems/ 3 implementations
├── 08_Stochastic_Statistical/   4 implementations
├── 09_Error_Analysis_Stability/ 3 implementations
├── 10_Quantum_Methods/          2 implementations
├── 11_Fluid_Dynamics/           2 implementations
├── 12_Particle_Methods/         2 implementations
├── 13_Signal_Processing/        1 implementation
├── 14_Automatic_Differentiation/1 implementation
├── 15_Interface_Methods/        2 implementations
├── 16_Advanced_Techniques/      2 implementations
│
└── README.md                   Learning roadmap & method catalog
```

---

## Module Dependency Graph

```
                    ┌──────────┐
                    │ main.py  │  ← entry point, mode detection
                    └────┬─────┘
                   ┌─────┴─────┐
              ┌────▼───┐  ┌────▼───┐
              │ cli.py │  │ gui.py │  ← user interfaces
              └──┬──┬──┘  └┬──┬──┬─┘
                 │  │      │  │  │
    ┌────────────┘  │      │  │  └────────────┐
    │               │      │  │               │
┌───▼────┐   ┌─────▼──────▼──┘          ┌────▼─────┐
│catalog │   │    utils.py  │           │equations │
│  .py   │   │ (docstring,  │           │   .py    │
│        │   │  run_file)   │           │(EQUATIONS│
│CATEGOR-│   └──────┬───────┘           │is_math_  │
│IES,    │          │                   │line)     │
│LEARNING│   ┌──────▼───────┐           └──────────┘
│_PATH   │   │  config.py   │
└────────┘   │ (BASE_DIR,   │
             │  colours,    │
             │  fonts)      │
             └──────────────┘
```

### Import Rules

| Module         | Imports from                          |
|----------------|---------------------------------------|
| `config.py`    | *(none — leaf)*                       |
| `catalog.py`   | *(none — pure data)*                  |
| `equations.py` | *(none — pure data + helper)*         |
| `utils.py`     | `config`                              |
| `cli.py`       | `config`, `catalog`, `utils`          |
| `gui.py`       | `config`, `catalog`, `equations`, `utils` |
| `main.py`      | `cli`, `gui` (lazy imports)           |

No circular dependencies. Data modules (`catalog`, `equations`, `config`)
are pure and import nothing from the package.

---

## Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                    User Interaction                     │
├──────────────────────┬──────────────────────────────────┤
│       CLI Mode       │           GUI Mode               │
│  (keyboard menus)    │  (tkinter dark theme)            │
│                      │                                  │
│  Browse → Category   │  Browse/Learn/Search panels      │
│  Search → Keyword    │  Rich-text theory panel          │
│  Learn → 7 stages    │  LaTeX equation rendering        │
│  Run   → subprocess  │  Threaded subprocess execution   │
└──────────────────────┴──────────────────────────────────┘
           │                        │
           ▼                        ▼
    ┌──────────────────────────────────────┐
    │           Catalog Layer              │
    │  CATEGORIES:  16 dicts with files[]  │
    │  LEARNING_PATH:  7-stage ordering    │
    │  EQUATIONS:  58 methods → LaTeX[]    │
    └──────────────────────────────────────┘
           │                        │
           ▼                        ▼
    ┌──────────────────────────────────────┐
    │           File System                │
    │  01_Linear_Algebra/*.py              │
    │  02_Differential_Equations/*.py      │
    │  ...                                 │
    │  16_Advanced_Techniques/*.py         │
    └──────────────────────────────────────┘
```

---

## GUI Rendering Pipeline

```
  Source .py file
       │
       ▼
  get_file_docstring()          ← Extract module docstring
       │
       ▼
  render_rich_text()            ← Parse lines, apply tags
       │
       ├── Title lines (===)     → tag: "title"    (15pt bold blue)
       ├── Headings (---)        → tag: "heading"  (12pt bold green)
       ├── **Bold**              → tag: "subheading"(10pt bold yellow)
       ├── Math lines (∂∇∫∑)    → tag: "equation" (10pt purple)
       ├── Bullet points (-)     → tag: "bullet"   (10pt indented)
       ├── Where to start        → tag: "start"    (green background)
       └── Normal text           → tag: "normal"   (10pt default)
       │
       ▼
  EQUATIONS[filename]           ← Lookup LaTeX entries
       │
       ▼
  _render_latex(latex_str)      ← matplotlib Figure → PNG → PhotoImage
       │
       ▼
  desc_text.image_create()      ← Embed rendered image in ScrolledText
```

---

## Equation Verification

All **~98 equations** across 58 methods have been cross-verified:

| Category                  | Methods | Equations | Status |
|---------------------------|---------|-----------|--------|
| Linear Algebra            | 10      | 28        | ✓      |
| Differential Equations    | 9       | 28        | ✓      |
| Numerical Integration     | 4       | 9         | ✓      |
| Interpolation & Approx.   | 5       | 13        | ✓      |
| Root-Finding              | 4       | 11        | ✓      |
| Optimization              | 4       | 11        | ✓      |
| Numerical Linear Systems  | 3       | 10        | ✓      |
| Stochastic & Statistical  | 4       | 10        | ✓      |
| Error Analysis & Stability| 3       | 9         | ✓      |
| Quantum Methods           | 2       | 5         | ✓      |
| Fluid Dynamics            | 2       | 5         | ✓      |
| Particle Methods          | 2       | 5         | ✓      |
| Signal Processing         | 1       | 2         | ✓      |
| Automatic Differentiation | 1       | 3         | ✓      |
| Interface Methods         | 2       | 7         | ✓      |
| Advanced Techniques       | 2       | 6         | ✓      |

### Corrections Applied

| Method | Field | Was | Corrected To |
|--------|-------|-----|-------------|
| Euler Method | Local Truncation Error | h/2 · y″(ξ) | **h²/2** · y″(ξ) |

---

## Usage

```bash
# Entry points (all equivalent):
python launcher.py              # auto-detect mode
python launcher.py --gui        # force GUI
python launcher.py --cli        # force CLI
python -m launcher_app          # package entry point

# Import API (backward compatible):
from launcher import CATEGORIES, EQUATIONS, is_math_line
from launcher_app.equations import EQUATIONS
from launcher_app.catalog import CATEGORIES, LEARNING_PATH
```

---

## Dependencies

| Package    | Purpose                                  | Required |
|------------|------------------------------------------|----------|
| Python 3.8+| Runtime                                  | Yes      |
| NumPy      | Numerical method implementations         | Yes      |
| matplotlib | Plotting + LaTeX equation rendering      | Yes      |
| Pygments   | Syntax highlighting support              | Yes      |
| tkinter    | GUI (bundled with Python on most systems)| Optional |
| SciPy      | Some method implementations              | Optional |
