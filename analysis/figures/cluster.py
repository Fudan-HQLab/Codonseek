"""Clustered similarity matrix heatmap.

Reads `matrix.txt`, greedily clusters sequences by an identity
threshold, reorders the matrix, and saves `clustered_heatmap.png`.

If `matrix.txt` is missing the script exits gracefully with a message.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
plt.rcParams["font.family"] = "sans-serif"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FILE_PATH = os.path.join("outputs", "matrix.txt")
OUTPUT_PATH = os.path.join("outputs", "clustered_heatmap.png")
IDENTITY_THRESHOLD = 0.90
MATRIX_SIZE = 100
FIG_SIZE = (9, 6)
DPI = 300
CMAP = "coolwarm"
VMIN = None
VMAX = 1.0

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

if not os.path.exists(FILE_PATH):
    print(f"[cluster] SKIP: {FILE_PATH} not found")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

data = np.genfromtxt(FILE_PATH, skip_header=1, usecols=range(1, MATRIX_SIZE + 1))
if data.shape != (MATRIX_SIZE, MATRIX_SIZE):
    print(f"[cluster] SKIP: expected shape ({MATRIX_SIZE},{MATRIX_SIZE}), got {data.shape}")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Greedy clustering
# ---------------------------------------------------------------------------

clusters = []
representatives = []

for i in range(MATRIX_SIZE):
    assigned = False
    for cid, rep in enumerate(representatives):
        if data[i, rep] >= IDENTITY_THRESHOLD:
            clusters[cid].append(i)
            assigned = True
            break
    if not assigned:
        representatives.append(i)
        clusters.append([i])

print(f"[cluster] {len(clusters)} clusters found")

# ---------------------------------------------------------------------------
# Reorder & plot
# ---------------------------------------------------------------------------

order = []
for c in clusters:
    order.extend(c)
data_sorted = data[np.ix_(order, order)]

fig, ax = plt.subplots(figsize=FIG_SIZE)
im = ax.imshow(data_sorted, cmap=CMAP, aspect="equal",
               interpolation="nearest", vmin=VMIN, vmax=VMAX)
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Similarity Score", fontsize=10)

ax.set_title("Clustered Similarity Matrix", fontsize=14)
ax.set_xlabel("Column Index", fontsize=12)
ax.set_ylabel("Row Index", fontsize=12)

tick_positions = [0] + list(range(9, MATRIX_SIZE, 10))
tick_labels = [1] + list(range(10, MATRIX_SIZE + 1, 10))
ax.set_xticks(tick_positions)
ax.set_yticks(tick_positions)
ax.set_xticklabels(tick_labels)
ax.set_yticklabels(tick_labels)
ax.invert_yaxis()

start = 0
for c in clusters:
    size = len(c)
    rect = patches.Rectangle(
        (start - 0.5, start - 0.5), size, size,
        linewidth=2, edgecolor="black", facecolor="none",
    )
    ax.add_patch(rect)
    start += size

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"[cluster] saved: {OUTPUT_PATH}")