"""Similarity matrix heatmap.

Reads `matrix.txt` and saves `heatmap.png`.
Gracefully exits if the input file is missing.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
plt.rcParams["font.family"] = "sans-serif"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FILE_PATH = os.path.join("outputs", "matrix.txt")
OUTPUT_PATH = os.path.join("outputs", "heatmap.png")
CMAP = "coolwarm"
VMIN = None
VMAX = 1.0
FIG_SIZE = (9, 6)
DPI = 300

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

if not os.path.exists(FILE_PATH):
    print(f"[heap_map] SKIP: {FILE_PATH} not found")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

data = np.genfromtxt(FILE_PATH, skip_header=1, usecols=range(1, 101))
if data.shape != (100, 100):
    print(f"[heap_map] SKIP: expected (100,100), got {data.shape}")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=FIG_SIZE)
im = ax.imshow(data, cmap=CMAP, aspect="equal", interpolation="nearest",
               vmin=VMIN, vmax=VMAX)
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Similarity Score", fontsize=10)

ax.set_xlabel("Column Index", fontsize=12)
ax.set_ylabel("Row Index", fontsize=12)
ax.set_title("Similarity Matrix", fontsize=14)

tick_positions = [0] + list(range(9, 100, 10))
tick_labels = [1] + list(range(10, 101, 10))
ax.set_xticks(tick_positions)
ax.set_yticks(tick_positions)
ax.set_xticklabels(tick_labels)
ax.set_yticklabels(tick_labels)
ax.invert_yaxis()
ax.grid(False)

plt.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"[heap_map] saved: {OUTPUT_PATH}")