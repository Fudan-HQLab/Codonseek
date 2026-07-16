"""Score density: first 200 vs last 200 self-play episodes.

Reads `RLscore/RLscore{1,2,3}.txt`, merges MCTS + selfplay scores
per step, then plots kernel-density estimates.

If any input file is missing the script exits gracefully.
"""

import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
plt.rcParams["font.family"] = "sans-serif"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_FILES = [
    os.path.join("outputs", "RLscore", "RLscore1.txt"),
    os.path.join("outputs", "RLscore", "RLscore2.txt"),
    os.path.join("outputs", "RLscore", "RLscore3.txt"),
]
OUTPUT_PATH = os.path.join("outputs", "density_compare.png")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_single_file(path):
    mcts_groups = []
    selfplay_scores = []
    current_mcts = []
    pattern = re.compile(r"Score:\s*(-?\d+\.\d{4})(?:\s+from_mcts|\s+selfplay)", re.IGNORECASE)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            score = float(m.group(1))
            if "selfplay" in line.lower():
                mcts_groups.append(current_mcts.copy())
                selfplay_scores.append(score)
                current_mcts.clear()
            else:
                current_mcts.append(score)

    return mcts_groups, selfplay_scores


def merge_multiple_scores(file_paths):
    all_mcts = []
    all_self = []
    min_len = None

    for fp in file_paths:
        mg, ss = parse_single_file(fp)
        all_mcts.append(mg)
        all_self.append(ss)
        min_len = len(ss) if min_len is None else min(min_len, len(ss))

    if min_len == 0:
        raise ValueError("All files are empty")

    aligned_mcts = [g[:min_len] for g in all_mcts]
    aligned_self = [s[:min_len] for s in all_self]

    avg_scores = []
    for i in range(min_len):
        combined = []
        for g in aligned_mcts:
            combined.extend(g[i])
        combined.extend([aligned_self[j][i] for j in range(len(file_paths))])
        avg_scores.append(sum(combined) / len(combined))

    print(f"[distribution] merged {min_len} steps  "
          f"(range {min(avg_scores):.4f} .. {max(avg_scores):.4f})")
    return avg_scores


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_density(scores, output_path, xlim=(0, 4)):
    if len(scores) < 400:
        print(f"[distribution] SKIP: need >=400 records, got {len(scores)}")
        return

    first = scores[:200]
    last = scores[-200:]

    kde_first = gaussian_kde(first)
    kde_last = gaussian_kde(last)

    x_min, x_max = xlim
    x_range = np.linspace(x_min, x_max, 500)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_range, kde_first(x_range), color="#A20EC7", linewidth=1.5, label="Untrained")
    ax.plot(x_range, kde_last(x_range), color="#2E5A88", linewidth=1.5, label="Trained")
    ax.set_xlabel("Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Reward Distribution", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 0.8)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[distribution] saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    missing = [p for p in INPUT_FILES if not os.path.exists(p)]
    if missing:
        print(f"[distribution] SKIP: missing {missing}")
        sys.exit(0)

    scores = merge_multiple_scores(INPUT_FILES)
    plot_density(scores, OUTPUT_PATH)