"""Convert a scored FASTA file into a pretrain data-buffer pickle.

Input FASTA headers must contain `label=N` and `expectation=X.XX`
(e.g. the output of `generate_traindata.py`).

Usage
-----
    cd codonseek && python pretrain_start/generate_pretrainpkl.py <input.fasta> <output.pkl>
"""

import pickle
import re
import sys
from pathlib import Path

import numpy as np
from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.mapper import aa2codon

LABEL2VALUE = {"0": 0.0, "1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0}
CODON2AA = {codon: aa for aa, codons in aa2codon.items() for codon in codons}


def _extract_label(description: str):
    m_label = re.search(r"label=(\d+)", description)
    m_exp = re.search(r"expectation=([-+]?\d+\.?\d*)", description)
    label = m_label.group(1) if m_label else None
    expectation = float(m_exp.group(1)) if m_exp else None
    return label, expectation


def _build_prob_vector(codon_triplet: str, bias: float = 0.5):
    if len(codon_triplet) < 9:
        return None
    codons = [codon_triplet[i:i + 3] for i in range(0, 9, 3)]
    aas = [CODON2AA.get(c) for c in codons]
    if None in aas:
        return None
    spaces = [aa2codon[aa] for aa in aas]
    total = len(spaces[0]) * len(spaces[1]) * len(spaces[2])
    if total == 0:
        return None
    prob = np.ones(total, dtype=float)
    try:
        idx0 = spaces[0].index(codons[0])
        idx1 = spaces[1].index(codons[1])
        idx2 = spaces[2].index(codons[2])
        true_idx = idx0 * (len(spaces[1]) * len(spaces[2])) + idx1 * len(spaces[2]) + idx2
        prob[true_idx] = 1.0 + bias * total
    except ValueError:
        pass
    prob /= prob.sum()
    final = np.zeros(216, dtype=float)
    final[:len(prob)] = prob
    return final


def fasta_to_pretrain_pkl(fasta_path: str, output_path: str):
    data_buffer = []
    skipped = 0

    for record in SeqIO.parse(fasta_path, "fasta"):
        label_key, expectation = _extract_label(record.description)
        if label_key is None or expectation is None:
            skipped += 1
            continue

        value = LABEL2VALUE[label_key] + expectation
        seq = str(record.seq)
        steps = []
        probs = []
        trajectory = ""

        for i in range(0, len(seq), 9):
            triplet = seq[i:i + 9]
            if len(triplet) < 9:
                break
            trajectory += triplet
            steps.append(trajectory)
            p = _build_prob_vector(triplet)
            if p is not None:
                probs.append(p)

        for s, p in zip(steps, probs):
            data_buffer.append((s, p, value))

    data_dict = {"data_buffer": data_buffer, "iters": 1}
    with open(output_path, "wb") as f:
        pickle.dump(data_dict, f)

    if skipped:
        print(f"  (skipped {skipped} records without label/expectation)")
    print(f"Pretrain buffer saved to {output_path}  ({len(data_buffer)} entries)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pretrain_start/generate_pretrainpkl.py <input.fasta> <pretrain.pkl>")
        sys.exit(1)
    fasta_to_pretrain_pkl(sys.argv[1], sys.argv[2])