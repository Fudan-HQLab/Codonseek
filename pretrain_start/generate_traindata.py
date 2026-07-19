"""Generate synonymous-variant codon sequences with TEN scores.

1. Starting from a wild-type coding sequence, produce a user-specified
   number of synonymous-codon variants.
2. Embed every variant with CaLM and score it with TEN.
3. Save a FASTA file with `label=N|expectation=X.XX` in the header,
   ready for `generate_pretrainpkl.py`.

Usage
-----
    cd codonseek && python pretrain_start/generate_traindata.py
"""

import random
import sys
from pathlib import Path

import torch

# Allow importing from the project root regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calm import load_calm
from ten.model import TENReward
from rl.config import CONFIG
from rl.mapper import aa2codon

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ORIGINAL_SEQ = ("")
OUTPUT_FILE = ""
NUM_SAMPLES = 1000
TARGET_DIFF = 0.25
SEED = 42
BATCH_SIZE = 32  # CaLM embedding batch size (adjust based on GPU memory).


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _codon_to_aa(codon: str) -> str:
    for aa, codons in aa2codon.items():
        if codon.upper() in codons:
            return aa
    raise ValueError(f"Unknown codon: {codon}")


def _split_codons(seq: str) -> list[str]:
    return [seq[i:i + 3] for i in range(0, len(seq), 3)]


# ---------------------------------------------------------------------------
# Variant generation
# ---------------------------------------------------------------------------

def generate_variants(
    initial_codons: list[str],
    num_samples: int,
    target_diff: float,
) -> list[str]:
    """Generate synonymous-codon variant sequences (full strings).

    Returns a list of codon-sequence strings; the first entry is always
    the original.
    """
    seq_len = len(initial_codons)
    num_mutations = max(1, int(seq_len * target_diff))

    variants: list[str] = ["".join(initial_codons)]
    seen: set[str] = {variants[0]}

    attempts = 0
    max_attempts = num_samples * 20

    while len(variants) < num_samples and attempts < max_attempts:
        attempts += 1

        positions = random.sample(range(seq_len), num_mutations)
        new_seq = initial_codons.copy()

        for pos in positions:
            aa = _codon_to_aa(initial_codons[pos])
            choices = [c for c in aa2codon[aa] if c != initial_codons[pos]]
            if choices:
                new_seq[pos] = random.choice(choices)

        seq_str = "".join(new_seq)
        if seq_str not in seen:
            variants.append(seq_str)
            seen.add(seq_str)

    return variants


# ---------------------------------------------------------------------------
# Scoring (CaLM + TEN)
# ---------------------------------------------------------------------------

def score_sequences(sequences: list[str], calm, ten, device) -> list[tuple[float, float, int]]:
    """Return `(score, expectation, label)` for every sequence."""
    results = []
    n = len(sequences)

    for start in range(0, n, BATCH_SIZE):
        batch = sequences[start:start + BATCH_SIZE]
        embeddings = calm.embed_sequences(batch)                # (B, 768)
        scores, exps, labels = ten.score_batch(embeddings)      # (B,)
        for i in range(len(batch)):
            results.append((
                float(scores[i]),
                float(exps[i]),
                int(labels[i]),
            ))
        if (start + BATCH_SIZE) % 128 == 0:
            print(f"  scored {min(start + BATCH_SIZE, n)}/{n}")

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_scored_fasta(
    sequences: list[str],
    scores: list[tuple[float, float, int]],
    path: str,
) -> None:
    with open(path, "w") as f:
        for idx, seq in enumerate(sequences):
            score, exp, label = scores[idx]
            f.write(f">seq_{idx} | label={label} | expectation={exp:.6f}\n{seq}\n")
    print(f"Saved {len(sequences)} scored sequences to {path}")


def print_mutation_stats(variants: list[str], original: str) -> None:
    if len(variants) <= 1:
        return

    seq_len = len(original)
    freq = [0] * seq_len
    ratios = []

    for seq in variants[1:]:
        muts = 0
        for i in range(seq_len):
            if seq[i] != original[i]:
                freq[i] += 1
                muts += 1
        ratios.append(muts / seq_len)

    avg_ratio = sum(ratios) / len(ratios)
    coverage = sum(1 for f in freq if f > 0)
    print(f"\nMutation stats:")
    print(f"  Mean mutation ratio: {avg_ratio:.3f}")
    print(f"  Position coverage:   {coverage}/{seq_len}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Generate variants
    codons = _split_codons(ORIGINAL_SEQ)
    print(f"Original sequence: {len(ORIGINAL_SEQ)} bp, {len(codons)} codons")
    print(f"Target mutation:   {TARGET_DIFF * 100:.0f}%"
          f"  ({int(len(codons) * TARGET_DIFF)} codons/variant)")
    print(f"Desired samples:   {NUM_SAMPLES}")

    variants = generate_variants(codons, NUM_SAMPLES, TARGET_DIFF)
    print(f"Generated {len(variants)} variants")
    print_mutation_stats(variants, "".join(codons))

    # 2. Score with CaLM + TEN
    print("\nLoading CaLM …")
    calm = load_calm()
    print("Loading TEN …")
    ten = TENReward(CONFIG["ten_model_path"], device=str(device))

    print(f"Scoring {len(variants)} sequences (batch_size={BATCH_SIZE}) …")
    scores = score_sequences(variants, calm, ten, device)

    # Summary
    labels = [s[2] for s in scores]
    print(f"\nScore summary:")
    for lbl in range(5):
        cnt = labels.count(lbl)
        print(f"  label {lbl}: {cnt}  ({100 * cnt / len(labels):.1f}%)")

    # 3. Save
    save_scored_fasta(variants, scores, OUTPUT_FILE)
    print("\nDone.  Next:  python pretrain_start/generate_pretrainpkl.py"
          f" {OUTPUT_FILE} pretrain.pkl")


if __name__ == "__main__":
    main()
