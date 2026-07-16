"""Training loss and entropy curves.

Reads `RL_loss.txt` (comma-separated: total_loss, policy_loss,
value_loss, entropy), downsamples, optionally smooths, and saves
`loss_entropy_curve.png`.

Gracefully exits if the input file is missing.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
plt.rcParams["font.family"] = "sans-serif"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FILE_PATH = "RL_loss.txt"
OUTPUT_PATH = os.path.join("outputs", "loss_entropy_curve.png")
DOWNSAMPLE = 5
SMOOTH = True
WINDOW = 10

COLOR_LOSS = "#2E5A88"
COLOR_ENTROPY = "#B83227"

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

if not os.path.exists(FILE_PATH):
    print(f"[pvnloss] SKIP: {FILE_PATH} not found")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

total_loss = []
entropy = []

with open(FILE_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        total_loss.append(float(parts[0]))
        entropy.append(float(parts[3]))

total_loss = np.array(total_loss)[::DOWNSAMPLE]
entropy = np.array(entropy)[::DOWNSAMPLE]
steps = np.arange(len(total_loss))


def moving_average(data, window):
    return np.convolve(data, np.ones(window) / window, mode="valid")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

ax1.plot(steps, total_loss, color=COLOR_LOSS, alpha=0.4, label="Total Loss")
if SMOOTH and len(total_loss) > WINDOW:
    smooth = moving_average(total_loss, WINDOW)
    ax1.plot(np.arange(len(smooth)), smooth, color=COLOR_LOSS, linewidth=2, label="Smoothed")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Total Loss")
ax1.set_title("Training Loss")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend()

ax2.plot(steps, entropy, color=COLOR_ENTROPY, alpha=0.4, label="Entropy")
if SMOOTH and len(entropy) > WINDOW:
    smooth = moving_average(entropy, WINDOW)
    ax2.plot(np.arange(len(smooth)), smooth, color=COLOR_ENTROPY, linewidth=2, label="Smoothed")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Entropy")
ax2.set_title("Information Entropy")
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=600)
plt.close()
print(f"[pvnloss] saved: {OUTPUT_PATH}")