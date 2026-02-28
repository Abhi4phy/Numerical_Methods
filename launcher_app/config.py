"""
Configuration — paths, colour scheme, fonts, constants.
========================================================
Single source of truth for every tuneable parameter in
both the CLI and GUI launchers.
"""

import os

# ────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────
# BASE_DIR = Numerical_Methods/  (parent of launcher_app/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ────────────────────────────────────────────────────────
# GUI colour palette  (Catppuccin Mocha-inspired)
# ────────────────────────────────────────────────────────
BG       = "#1e1e2e"   # Base background
BG2      = "#282840"   # Secondary background (panels)
BG3      = "#313150"   # Tertiary / hover
FG       = "#cdd6f4"   # Primary foreground
FG_DIM   = "#7f849c"   # Dimmed text
ACCENT   = "#89b4fa"   # Blue  — titles, links
ACCENT2  = "#a6e3a1"   # Green — success, headings
ACCENT3  = "#f9e2af"   # Yellow — warnings, subheadings
RED      = "#f38ba8"   # Red   — errors, missing files
PURPLE   = "#cba6f7"   # Mauve — equations
PINK     = "#f5c2e7"   # Pink  — equation labels
ORANGE   = "#fab387"   # Peach — code blocks
GREEN_BG = "#1e3a2e"   # "Where to start" background
SEP_CLR  = "#45475a"   # Separator lines

# ────────────────────────────────────────────────────────
# Fonts
# ────────────────────────────────────────────────────────
FONT     = ("Consolas", 10)
FONT_B   = ("Consolas", 11, "bold")
FONT_H   = ("Consolas", 14, "bold")
FONT_SM  = ("Consolas", 9)
FONT_LG  = ("Consolas", 15, "bold")
FONT_EQ  = ("Consolas", 13, "bold")

# ────────────────────────────────────────────────────────
# Runtime limits
# ────────────────────────────────────────────────────────
SCRIPT_TIMEOUT = 120        # seconds
LATEX_DPI      = 110        # rendered equation resolution
LATEX_FIG_W    = 8          # figure width (inches)
LATEX_FIG_H    = 0.5        # figure height (inches)
