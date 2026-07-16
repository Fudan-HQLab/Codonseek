"""Base-composition stacked bars for top-100 sequences.

Reads `LBA.fasta`, computes A/U/C/G proportions for the first 100
sequences, and saves `stacked_bars.png`.

Gracefully exits if the input file is missing.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Bio import SeqIO

plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
plt.rcParams["font.family"] = "sans-serif"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FASTA_FILE = os.path.join("outputs", "LBA.fasta")
OUTPUT_PATH = os.path.join("outputs", "stacked_bars.png")
TOP_N = 100

COLORS = {"A": "#476f95", "U": "#7593af", "C": "#a3b7ca", "G": "#d1dbe4"}


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

if not os.path.exists(FASTA_FILE):
    print(f"[stacked_map] SKIP: {FASTA_FILE} not found")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def parse_top_n_fasta(path, top_n=TOP_N):
    bases = ["A", "U", "C", "G"]
    sequences = []
    for i, rec in enumerate(SeqIO.parse(path, "fasta")):
        if i >= top_n:
            break
        sequences.append(str(rec.seq).upper().replace("T", "U"))

    n_seq = len(sequences)
    comp = np.zeros((n_seq, len(bases)))
    for i, seq in enumerate(sequences):
        total = len(seq)
        if total == 0:
            continue
        counts = [seq.count(b) for b in bases]
        comp[i] = [c / total for c in counts]

    return bases, comp


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_stacked_bars(bases, comp, output_path=OUTPUT_PATH):
    n_seq = comp.shape[0]
    x = np.arange(n_seq)

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(n_seq)
    for i, base in enumerate(bases):
        ax.bar(x, comp[:, i], width=1.0, bottom=bottom,
               label=base, color=COLORS[base], edgecolor=None, linewidth=0)
        bottom += comp[:, i]

    ax.set_xlabel("Sequence Rank", fontsize=14)
    ax.set_ylabel("Proportion", fontsize=14)
    ax.set_title("Base Composition of Top 100 Sequences", fontsize=18)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, n_seq - 0.5)

    tick_pos = np.arange(0, n_seq + 10, 10)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_pos)
    ax.tick_params(axis="x", length=0)

    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[stacked_map] saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bases, comp = parse_top_n_fasta(FASTA_FILE)
    if comp.shape[0] == 0:
        print("[stacked_map] SKIP: no sequences found")
        sys.exit(0)
    print(f"[stacked_map] matrix: {comp.shape}  (avg: {comp.mean(axis=0)})")
    plot_stacked_bars(bases, comp, OUTPUT_PATH)