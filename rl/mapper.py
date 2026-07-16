"""Codon sequence processing utilities and lookup tables.

Provides:
  ``aa2codon``       – amino acid → synonymous codon list
  ``codon_matrix``   – codon → (4 × 3) one-hot encoding for CNN input
  ``Sequencematrix`` – encode a sequence string as a CNN-ready tensor
  ``CodonMapper``    – legal-move generation for MCTS
"""

import itertools
import numpy as np
from rl.config import CONFIG


# ===========================================================================
# Genetic code: amino acid → synonymous codons
# ===========================================================================

aa2codon = {
    "A": ["GCU", "GCC", "GCA", "GCG"],
    "R": ["CGU", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "N": ["AAU", "AAC"],
    "D": ["GAU", "GAC"],
    "C": ["UGU", "UGC"],
    "Q": ["CAA", "CAG"],
    "E": ["GAA", "GAG"],
    "G": ["GGU", "GGC", "GGA", "GGG"],
    "H": ["CAU", "CAC"],
    "I": ["AUU", "AUC", "AUA"],
    "L": ["CUU", "CUC", "CUA", "CUG", "UUA", "UUG"],
    "K": ["AAA", "AAG"],
    "M": ["AUG"],
    "F": ["UUU", "UUC"],
    "P": ["CCU", "CCC", "CCA", "CCG"],
    "S": ["UCU", "UCC", "UCA", "UCG", "AGU", "AGC"],
    "T": ["ACU", "ACC", "ACA", "ACG"],
    "W": ["UGG"],
    "Y": ["UAU", "UAC"],
    "V": ["GUU", "GUC", "GUA", "GUG"],
    "*": ["UAA", "UAG", "UGA"],
}


# ===========================================================================
# Codon one-hot encoding matrix (4 rows = A/C/G/U, 3 columns per codon)
# ===========================================================================

codon_matrix = {
'GCU': [[0., 0., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [1., 0., 0.]],

'GCC': [[0., 0., 0.],
       [0., 0., 0.],
       [0., 1., 1.],
       [1., 0., 0.]],

'GCA': [[0., 0., 1.],
       [0., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.]],

'GCG': [[0., 0., 0.],
       [0., 0., 0.],
       [0., 1., 0.],
       [1., 0., 1.]],

'CGU': [[0., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.]],

'CGC': [[0., 0., 0.],
       [0., 0., 0.],
       [1., 0., 1.],
       [0., 1., 0.]],

'CGA': [[0., 0., 1.],
       [0., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.]],

'CGG': [[0., 0., 0.],
       [0., 0., 0.],
       [1., 0., 0.],
       [0., 1., 1.]],

'AGA': [[1., 0., 1.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 1., 0.]],

'AGG': [[1., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 1., 1.]],

'AAU': [[1., 1., 0.],
       [0., 0., 0.],
       [0., 0., 1.],
       [0., 0., 0.]],

'AAC': [[1., 1., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 1.]],

'GAU': [[0., 1., 0.],
       [0., 0., 1.],
       [0., 0., 0.],
       [1., 0., 0.]],

'GAC': [[0., 1., 0.],
       [0., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.]],

'UGU': [[0., 0., 0.],
       [1., 0., 1.],
       [0., 0., 0.],
       [0., 1., 0.]],

'UGC': [[0., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [0., 1., 0.]],

'CAA': [[0., 1., 1.],
       [0., 0., 0.],
       [1., 0., 0.],
       [0., 0., 0.]],

'CAG': [[0., 1., 0.],
       [0., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.]],

'GAA': [[0., 1., 1.],
       [0., 0., 0.],
       [0., 0., 0.],
       [1., 0., 0.]],

'GAG': [[0., 1., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [1., 0., 1.]],

'GGU': [[0., 0., 0.],
       [0., 0., 1.],
       [0., 0., 0.],
       [1., 1., 0.]],

'GGC': [[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 1.],
       [1., 1., 0.]],

'GGA': [[0., 0., 1.],
       [0., 0., 0.],
       [0., 0., 0.],
       [1., 1., 0.]],

'GGG': [[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [1., 1., 1.]],

'CAU': [[0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 0., 0.]],

'CAC': [[0., 1., 0.],
       [0., 0., 0.],
       [1., 0., 1.],
       [0., 0., 0.]],

'AUU': [[1., 0., 0.],
       [0., 1., 1.],
       [0., 0., 0.],
       [0., 0., 0.]],

'AUC': [[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 0., 0.]],

'AUA': [[1., 0., 1.],
       [0., 1., 0.],
       [0., 0., 0.],
       [0., 0., 0.]],

'CUU': [[0., 0., 0.],
       [0., 1., 1.],
       [1., 0., 0.],
       [0., 0., 0.]],

'CUC': [[0., 0., 0.],
       [0., 1., 0.],
       [1., 0., 1.],
       [0., 0., 0.]],

'CUA': [[0., 0., 1.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 0.]],

'CUG': [[0., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.]],

'UUA': [[0., 0., 1.],
       [1., 1., 0.],
       [0., 0., 0.],
       [0., 0., 0.]],

'UUG': [[0., 0., 0.],
       [1., 1., 0.],
       [0., 0., 0.],
       [0., 0., 1.]],

'AAA': [[1., 1., 1.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]],

'AAG': [[1., 1., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 1.]],

'AUG': [[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 0.],
       [0., 0., 1.]],

'UUU': [[0., 0., 0.],
       [1., 1., 1.],
       [0., 0., 0.],
       [0., 0., 0.]],

'UUC': [[0., 0., 0.],
       [1., 1., 0.],
       [0., 0., 1.],
       [0., 0., 0.]],

'CCU': [[0., 0., 0.],
       [0., 0., 1.],
       [1., 1., 0.],
       [0., 0., 0.]],

'CCC': [[0., 0., 0.],
       [0., 0., 0.],
       [1., 1., 1.],
       [0., 0., 0.]],

'CCA': [[0., 0., 1.],
       [0., 0., 0.],
       [1., 1., 0.],
       [0., 0., 0.]],

'CCG': [[0., 0., 0.],
       [0., 0., 0.],
       [1., 1., 0.],
       [0., 0., 1.]],

'UCU': [[0., 0., 0.],
       [1., 0., 1.],
       [0., 1., 0.],
       [0., 0., 0.]],

'UCC': [[0., 0., 0.],
       [1., 0., 0.],
       [0., 1., 1.],
       [0., 0., 0.]],

'UCA': [[0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 0.]],

'UCG': [[0., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]],

'AGU': [[1., 0., 0.],
       [0., 0., 1.],
       [0., 0., 0.],
       [0., 1., 0.]],

'AGC': [[1., 0., 0.],
       [0., 0., 0.],
       [0., 0., 1.],
       [0., 1., 0.]],

'ACU': [[1., 0., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 0., 0.]],

'ACC': [[1., 0., 0.],
       [0., 0., 0.],
       [0., 1., 1.],
       [0., 0., 0.]],

'ACA': [[1., 0., 1.],
       [0., 0., 0.],
       [0., 1., 0.],
       [0., 0., 0.]],

'ACG': [[1., 0., 0.],
       [0., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]],

'UGG': [[0., 0., 0.],
       [1., 0., 0.],
       [0., 0., 0.],
       [0., 1., 1.]],

'UAU': [[0., 1., 0.],
       [1., 0., 1.],
       [0., 0., 0.],
       [0., 0., 0.]],

'UAC': [[0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [0., 0., 0.]],

'GUU': [[0., 0., 0.],
       [0., 1., 1.],
       [0., 0., 0.],
       [1., 0., 0.]],

'GUC': [[0., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.]],

'GUA': [[0., 0., 1.],
       [0., 1., 0.],
       [0., 0., 0.],
       [1., 0., 0.]],

'GUG': [[0., 0., 0.],
       [0., 1., 0.],
       [0., 0., 0.],
       [1., 0., 1.]],

'UAA': [[0., 1., 1.],
       [1., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]],

'UAG': [[0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 0.],
       [0., 0., 1.]],

'UGA': [[0., 0., 1.],
       [1., 0., 0.],
       [0., 0., 0.],
       [0., 1., 0.]],
}


# ===========================================================================
# Sequence encoding
# ===========================================================================

def Sequencematrix(codon_sequence):
    """Convert a codon nucleotide sequence to a (1, 4, max_len) one-hot tensor.

    Each codon triplet is expanded into three 4-bit one-hot columns
    (rows = A/C/G/U), producing a matrix suitable as input to the CNN.
    """
    max_len = len(CONFIG["Protein_sequence"]) * 3
    codons = [codon_sequence[i:i + 3].upper() for i in range(0, len(codon_sequence), 3)]
    encoded_seq = np.zeros((4, max_len))
    for idx, codon in enumerate(codons):
        matrix = codon_matrix[codon]
        start_col = 3 * idx
        encoded_seq[:, start_col:start_col + 3] = matrix
    return encoded_seq[np.newaxis, :, :]


# ===========================================================================
# Legal-move generation for MCTS
# ===========================================================================

class CodonMapper:
    """Build the mapping from a codon-sequence prefix to the set of legal
    next-codon choices.

    Given the target protein sequence (from ``CONFIG["Protein_sequence"]``),
    this class determines which codons are allowed at each amino-acid
    position and produces the full legal-move vector + candidate
    sequences for MCTS.
    """

    def __init__(self, initial_sequence: str):
        self.codon_sequence = initial_sequence
        self.codon_dict = {}          # global_index -> (amino_acid_position, codon_index)
        self.position_range_dict = {} # amino_acid_position -> (start_index, end_index)
        self._build_mapping()

    def _build_mapping(self):
        """Populate ``codon_dict`` and ``position_range_dict``."""
        self.codon_dict.clear()
        self.position_range_dict.clear()
        current_num = 1
        protein_sequence = CONFIG["Protein_sequence"]

        for aa_pos, residue in enumerate(protein_sequence):
            if residue == "*":
                self.position_range_dict[aa_pos] = None
                continue

            codons = aa2codon.get(residue)
            if not codons:
                raise ValueError(f"Unknown residue '{residue}' at position {aa_pos}")

            start = current_num
            end = current_num + len(codons) - 1
            self.position_range_dict[aa_pos] = (start, end)

            for codon_idx in range(len(codons)):
                self.codon_dict[current_num] = (aa_pos, codon_idx)
                current_num += 1

    def available_codon(self, sequence: str):
        """Return (legal_moves_vector, candidate_sequences) for *sequence*.

        Parameters
        ----------
        sequence : str
            The current codon-sequence prefix (e.g. "AUGCUGGAA...").

        Returns
        -------
        vector : np.ndarray of shape (216,) float32
            Binary mask where 1.0 means the corresponding codon combination
            is legal at this step.
        full_next_seq : list of str
            All candidate sequences resulting from appending one of the
            legal codon combinations to *sequence*.
        """
        protein_sequence = CONFIG["Protein_sequence"]
        completed_aa_count = len(sequence) // 3

        # Look ahead up to 3 amino acids (max 4 codons each = 216 combinations)
        next_aas = protein_sequence[completed_aa_count:completed_aa_count + 3]
        num_aas = len(next_aas)

        choices_per_pos = []
        for aa in next_aas:
            choices = aa2codon.get(aa, [])
            if not choices:
                print(f"Warning: amino acid '{aa}' has no codon.")
            choices_per_pos.append(choices)

        all_combinations = list(itertools.product(*choices_per_pos))
        num_combinations = len(all_combinations)

        full_next_seq = []
        for combo in all_combinations:
            padded = list(combo) + [""] * (3 - num_aas)
            new_fragment = "".join(padded)
            full_next_seq.append(sequence + new_fragment)

        vector = np.zeros(216, dtype=np.float32)
        vector[:num_combinations] = 1
        return vector, full_next_seq


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seq = "AUGGCC"
    mapper = CodonMapper(seq)
    vec, candidates = mapper.available_codon(seq)
    print("legal vector shape:", vec.shape)
    print("num candidates:", len(candidates))
    print("example:", candidates[:3])
    print("Sequencematrix shape:", Sequencematrix(seq).shape)

