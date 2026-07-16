"""MCTS (Monte Carlo Tree Search) for codon optimisation.

Provides `MCTS` (the search engine) and `MCTSPlayer` (a turn-based
player wrapper that calls MCTS and selects a move).
"""

import heapq
import math
import os
import time

import numpy as np
import torch

from calm import load_calm
from ten.model import TENReward
from rl.config import CONFIG
from rl.mapper import CodonMapper
from rl._utils import timer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def softmax(x: np.ndarray) -> np.ndarray:
    """Stable softmax over a 1-D array."""
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


# ---------------------------------------------------------------------------
# TreeNode -- tree structure preserved from the reference implementation.
# ---------------------------------------------------------------------------


class TreeNode(object):

    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children: dict = {}
        self.n_visits = 0
        self.Q = 0.0
        self.u = 0.0
        self.p = prior_p

    def expand(self, act_probs):
        """Create child nodes from (sequence, prob) pairs."""
        for sequence, prob in act_probs:
            if sequence not in self.children:
                self.children[sequence] = TreeNode(self, prob)

    def select(self, c_puct):
        """Return (sequence, child_node) with the highest Q+U."""
        return max(
            self.children.items(),
            key=lambda kv: kv[1].get_value(c_puct),
        )

    def get_value(self, c_puct):
        """Compute Q + U (UCB value) for this node."""
        self.u = (
            c_puct
            * self.p
            * np.sqrt(self.parent.n_visits)
            / (1 + self.n_visits)
        )
        return self.Q + self.u

    def update(self, leaf_value):
        """Incremental averaging of Q over visits."""
        self.n_visits += 1
        self.Q += (leaf_value - self.Q) / self.n_visits

    def update_recursive(self, leaf_value):
        """Backpropagate *leaf_value* up to the root."""
        if self.parent is not None:
            self.parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

# Module-level CaLM cache -- shared across all MCTS instances in this process.
_CALM = load_calm()


class MCTS(object):
    """Monte Carlo Tree Search engine for codon-sequence optimisation.

    Parameters
    ----------
    policy_value_fn :
        Callable `(codon_seq) -> (seek_probs, leaf_value)` that returns
        prior probabilities and a value estimate for a given sequence prefix.
    ten_model_path :
        Path to a TEN checkpoint, used to evaluate full-length sequences.
    collect_index :
        Process index for output-file naming.
    c_puct :
        Exploration constant.
    n_playout :
        Number of playouts per move.
    """

    def __init__(
        self,
        policy_value_fn,
        ten_model_path: str,
        collect_index: int,
        c_puct: float = 0,
        n_playout: int = 200,
    ):
        self.reward_model = TENReward(ten_model_path, device="cuda:0")
        self.idx = collect_index
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

        self.playout_times = 0
        # Each entry: (score, label, exp_val, seq)
        self.top_heap: list[tuple[float, int, float, str]] = []
        self.existing_seqs: set[str] = set()

        os.makedirs("./outputs/mcts_top_heap", exist_ok=True)
        os.makedirs("./outputs/RLscore", exist_ok=True)
        self.file_name = f"./outputs/mcts_top_heap/mcts_top_heap{self.idx}.txt"

        # Restore existing_seqs from any previously persisted top-heap file
        # so that a fresh MCTS (created after policy reload) does not lose
        # track of sequences already written by the previous MCTS instance.
        self._load_existing_from_file()

        # Derived constants
        self._protein_len = len(CONFIG["Protein_sequence"])
        self._max_len = 3 * self._protein_len
        self._cutoff = int(self._max_len * 0.75)

    # ------------------------------------------------------------------
    # Embedding & evaluation
    # ------------------------------------------------------------------

    def _embed(self, sequence: str) -> torch.Tensor:
        """Return a `(1, 768)` CaLM embedding for *sequence* on GPU."""
        vector = _CALM.embed_sequence(sequence)
        if isinstance(vector, list):
            vector = np.array(vector)
        if isinstance(vector, np.ndarray):
            vector = torch.from_numpy(vector).float()
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        return vector.to(self.reward_model.device)

    def _evaluate(self, sequence: str) -> tuple[float, float, int]:
        """Score *sequence* via TEN reward model, write result to RLscore log.

        Returns (score, expectation, label).
        """
        state_tensor = self._embed(sequence)
        score, exp_val, label = self.reward_model.score(state_tensor)

        with open(f"./outputs/RLscore/RLscore{self.idx}.txt", "a") as f:
            f.write(f"Score: {score:.4f} from_mcts\n")
        return score, exp_val, label

    # ------------------------------------------------------------------
    # Tree search
    # ------------------------------------------------------------------

    def _playout(self, codon_sequence: str) -> int:
        """Run one MCTS playout from *codon_sequence*.

        Returns 1 if the TEN reward model was invoked at the leaf, 0 otherwise.
        """
        node = self._root
        current = codon_sequence

        # Descend to a leaf
        while not node.is_leaf():
            current, node = node.select(self._c_puct)

        cur_len = len(current)
        use_ten = cur_len >= self._cutoff

        if use_ten:
            score, exp_val, label = self._evaluate(current)
            leaf_value = score
            used_ten = 1

            if cur_len == self._max_len:
                self._update_top_scores(label, exp_val, current)
                self._save_top_heap()
        else:
            _, leaf_value = self._policy(current)
            used_ten = 0

        # Expand (unless already at terminal length)
        if cur_len < self._max_len:
            legal_moves, candidate_sequences = CodonMapper(current).available_codon(current)
            act_probs, _ = self._policy(current)
            node.expand(act_probs)

        node.update_recursive(leaf_value)
        return used_ten

    # ------------------------------------------------------------------
    # Move-probability computation
    # ------------------------------------------------------------------

    @timer("get_move_probs")
    def get_move_probs(self, sequence: str, temp: float = 1e-3):
        """Run *n_playout* simulations and return `(seqs, probs)`."""
        ten_count = 0
        t0 = time.perf_counter()

        for _ in range(self._n_playout):
            ten_count += self._playout(sequence)

        self.playout_times += 1
        elapsed = time.perf_counter() - t0
        print(f"playout_times: {self.playout_times}  "
              f"ten_predict_count: {ten_count}  "
              f"Running Time: {elapsed:.2f} s")


        # Build move probabilities from root visit counts
        act_visits = [
            (seq, node.n_visits)
            for seq, node in self._root.children.items()
        ]
        sequences, visits = zip(*act_visits)
        visits_arr = np.array(visits, dtype=np.float32)
        move_probs = softmax((1.0 / temp) * np.log(visits_arr + 1e-10))

        return sequences, move_probs

    # ------------------------------------------------------------------
    # Tree pruning (after a real move)
    # ------------------------------------------------------------------

    def update_with_move(self, sequence: str) -> None:
        """Advance the root to *sequence* if it exists in the tree."""
        if sequence in self._root.children:
            self._root = self._root.children[sequence]
            self._root.parent = None
        else:
            self._root = TreeNode(None, 1.0)

    # ------------------------------------------------------------------
    # Top-k heap management
    # ------------------------------------------------------------------

    def _load_existing_from_file(self) -> None:
        """Populate `existing_seqs` from any previously persisted heap file.

        This ensures a newly created MCTS (e.g. after a policy reload during
        collect) does not overwrite or lose records written by the previous
        MCTS instance that shared the same output file.
        """
        try:
            with open(self.file_name, "r") as f:
                for line in f:
                    # Format: "Label: 3 | Expectation: 0.1234 | Seq: ATG..."
                    if "| Seq: " in line:
                        seq = line.rsplit("| Seq: ", 1)[-1].strip()
                        if seq:
                            self.existing_seqs.add(seq)
        except OSError:
            pass

    def _update_top_scores(self, label: int, exp_val: float, seq: str) -> None:
        """Insert *seq* into the top-score heap, deduplicating by sequence.

        Score = label + exp_val.
        """
        if seq in self.existing_seqs:
            return
        score = float(label) + exp_val
        item = (score, label, exp_val, seq)
        self.existing_seqs.add(seq)
        self.top_heap.append(item)
        self.top_heap.sort(key=lambda x: (x[0], x[1]))

    def _save_top_heap(self) -> None:
        """Persist the sorted top heap to disk, merging with any records that
        already exist in the file (from a previous MCTS instance).

        Format:  Label: {label_int} | Expectation: {exp_val} | Seq: {seq}
        Score = Label + Expectation.
        """
        if not self.top_heap:
            return

        # Merge with existing file contents so that records written by a
        # previous MCTS instance (before a policy reload) are not lost.
        existing_items: list[tuple[float, int, float, str]] = []
        try:
            with open(self.file_name, "r") as f:
                for line in f:
                    if "| Seq: " not in line:
                        continue
                    try:
                        parts = line.split("|")
                        label_str = parts[0].replace("Label:", "").strip()
                        exp_str = parts[1].replace("Expectation:", "").strip()
                        seq = parts[2].replace("Seq:", "").strip()
                        lbl = int(label_str)
                        exp_val = float(exp_str)
                        scr = float(lbl) + exp_val
                        existing_items.append((scr, lbl, exp_val, seq))
                    except (ValueError, IndexError):
                        continue
        except OSError:
            pass

        # Merge, deduplicate by sequence, keep highest score
        all_items: dict[str, tuple[float, int, float]] = {}
        for score, lbl, exp_val, seq in existing_items:
            if seq not in all_items or score > all_items[seq][0]:
                all_items[seq] = (score, lbl, exp_val)

        for score, lbl, exp_val, seq in self.top_heap:
            if seq not in all_items or score > all_items[seq][0]:
                all_items[seq] = (score, lbl, exp_val)

        merged = sorted(
            [(s, l, e, seq) for seq, (s, l, e) in all_items.items()],
            key=lambda x: (x[0], x[1]),
        )

        with open(self.file_name, "w") as f:
            for score, label, exp_val, seq in merged:
                f.write(
                    f"Label: {label} | Expectation: {exp_val:.4f} | Seq: {seq}\n"
                )

    def __str__(self):
        return "MCTS"


# ---------------------------------------------------------------------------
# MCTSPlayer
# ---------------------------------------------------------------------------


class MCTSPlayer(object):
    """A turn-based agent wrapping an MCTS engine.

    `get_action` runs a full MCTS search from the current sequence
    prefix and returns the selected next sequence plus a 216-dim
    policy-vector for training the PVN.
    """

    def __init__(
        self,
        policy_value_function,
        ten_model_path: str,
        collect_index: int,
        c_puct: float = 5,
        n_playout: int = 1000,
        is_selfplay: int = 0,
    ):
        self.mcts = MCTS(
            policy_value_fn=policy_value_function,
            ten_model_path=ten_model_path,
            collect_index=collect_index,
            c_puct=c_puct,
            n_playout=n_playout,
        )
        self._is_selfplay = is_selfplay
        self.agent = "AI"

    def __str__(self):
        return f"MCTS {self.agent}"

    @timer("get_action")
    def get_action(self, seq: str):
        """Run MCTS search and return `(next_sequence, policy_vector)`.

        The policy vector is a 216-dim `ndarray` whose non-zero entries
        correspond to legal moves with their MCTS visit-probabilities.
        """
        legal_moves, candidate_sequences = CodonMapper(seq).available_codon(seq)
        indices = np.where(legal_moves == 1)[0]

        seek_sequences, move_probs = self.mcts.get_move_probs(seq, temp=0.5)

        # Build a mapping from candidate sequence -> its MCTS probability.
        # Some legal moves may not have been visited; those get zero probability.
        seq_to_prob = dict(zip(seek_sequences, move_probs))
        full_probs = np.array(
            [seq_to_prob.get(s, 0.0) for s in candidate_sequences],
            dtype=np.float64,
        )
        full_probs /= full_probs.sum()  # re-normalise

        # Move selection with Dirichlet noise for exploration
        noise = np.random.dirichlet(
            CONFIG["dirichlet"] * np.ones(len(full_probs))
        )
        blended = 0.70 * full_probs + 0.30 * noise
        move = np.random.choice(indices, p=blended)
        chosen_seq = candidate_sequences[list(indices).index(move)]

        # Build the 216-dim policy vector
        policy_vector = np.zeros(216, dtype=np.float32)
        policy_vector[indices] = full_probs

        self.mcts.update_with_move(chosen_seq)
        return chosen_seq, policy_vector
