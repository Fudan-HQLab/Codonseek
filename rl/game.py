"""Self-play game engine for codon optimisation.

`SelfSampling` orchestrates a complete self-play episode:
given a starting codon-sequence prefix it repeatedly asks the MCTS
player for a move, evaluates the resulting sequence with the
TEN reward model, and accumulates training data (sequence, policy
vector, score) for the policy-value network.

Once the full-length codon sequence is assembled the episode
finishes and the collected data is yielded to the caller.
"""

import heapq
import os
import time

import numpy as np
import torch

from calm import load_calm
from ten.model import TENReward
from rl.config import CONFIG
from rl._utils import timer


# ---------------------------------------------------------------------------
# SelfSampling
# ---------------------------------------------------------------------------

class SelfSampling:
    """Runs self-play episodes and keeps a running top-k heap of best sequences.

    Parameters
    ----------
    device :
        PyTorch device string.  Defaults to `"cuda:0"` (falls back to
        `"cpu"` if CUDA is unavailable).
    top_heap_max_keep :
        Maximum number of sequences retained in the top-score heap.
    """

    def __init__(self, device: str = "cuda:0", top_heap_max_keep: int = 100):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_keep = top_heap_max_keep

        # Evaluation models
        self.reward_model = TENReward(CONFIG["ten_model_path"], device=self.device)
        self.calm_model = load_calm()

        # Per-episode tracking
        self.selftimes = 0
        self.top_heap: list[tuple[float, int, str]] = []
        self.existing_seqs: set[str] = set()
        self.eval_error_count = 0

        # Ensure output directories exist before any writes.
        os.makedirs("./outputs/RLscore", exist_ok=True)
        os.makedirs("./outputs/selfplay_top_heap", exist_ok=True)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, sequence: str) -> torch.Tensor:
        """Return a `(1, 768)` CaLM embedding for *sequence* on the target device."""
        vector = self.calm_model.embed_sequence(sequence)

        # `embed_sequence` returns a tensor; the guards below are kept
        # for backwards-compatibility in case of future API changes.
        if isinstance(vector, list):
            vector = np.array(vector)
        if isinstance(vector, np.ndarray):
            vector = torch.from_numpy(vector).float()
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        return vector.to(self.device)

    # ------------------------------------------------------------------
    # Evaluation (TEN reward)
    # ------------------------------------------------------------------

    def _evaluate(self, sequence: str) -> tuple[float, float, int]:
        """Score a single codon sequence with the TEN reward model.

        Returns
        -------
        score : float
            Softmax label + weighted-probability sum (for training signal).
        exp_val : float
            Weighted-probability expectation (values approx [-1, 1]).
        label : int
            Hard class index (0 -- 4).
        """
        try:
            input_tensor = self._embed(sequence)
            with torch.no_grad():
                score, exp_val, label = self.reward_model.score(input_tensor)
            return score, exp_val, label
        except Exception as exc:
            self.eval_error_count += 1
            if self.eval_error_count <= 5:
                print(f"Evaluation error: {exc}")
            return 0.0, 0.0, 0

    # ------------------------------------------------------------------
    # Self-play episode
    # ------------------------------------------------------------------

    @timer("self_play")
    def self_play(self, sequence, player, index):
        """Run one complete self-play episode.

        Parameters
        ----------
        sequence :
            Starting codon-sequence prefix (usually `""`).
        player :
            An `MCTSPlayer` instance that provides `get_action(seq)`.
        index :
            Collect-process index (used for output-file naming).

        Yields
        ------
        (new_seq, mcts_prob, score)  for each step of the episode.
        """
        current_seq = sequence if sequence else ""
        new_sequences: list[str] = []
        mcts_probs: list[np.ndarray] = []
        all_scores: list[float] = []
        step = 0

        protein_len = len(CONFIG["Protein_sequence"])
        max_len = 3 * protein_len
        cutoff = int(max_len * 0.75)

        last_score = 0.0
        last_exp_val = 0.0
        last_label = 0

        while True:
            new_seq, mcts_prob = player.get_action(current_seq)
            new_sequences.append(new_seq)
            mcts_probs.append(mcts_prob)
            step += 1
            current_seq = new_seq
            cur_len = len(current_seq)

            print(f"Step {step:3d}  len={cur_len:3d}  {current_seq}")

            if cur_len > cutoff:
                score, exp_val, label = self._evaluate(new_seq)
                all_scores.append(score)
                last_label, last_exp_val, last_score = label, exp_val, score

            if cur_len >= max_len:
                # Compute per-step scores for training: prefix steps get
                # the average terminal score so the PVN sees smooth signals.
                all_scores_arr = np.array(all_scores)
                avg_score = float(np.mean(all_scores_arr)) if all_scores_arr.size else 0.0
                score_prefix = np.full(step, avg_score, dtype=np.float32)
                scores = np.concatenate([score_prefix, all_scores_arr], axis=0)

                self._update_top_scores(last_label, last_exp_val, new_seq)
                self._save_top_heap(index)

                with open(f"./outputs/RLscore/RLscore{index}.txt", "a") as f:
                    f.write(f"Score: {last_score:.4f} selfplay: {self.selftimes + 1}\n")
                self.selftimes += 1

                return zip(new_sequences, mcts_probs, scores)

    # ------------------------------------------------------------------
    # Top-k heap management
    # ------------------------------------------------------------------

    def reset_heap(self) -> None:
        """Clear the top-score heap."""
        self.top_heap.clear()
        self.existing_seqs.clear()

    def _update_top_scores(self, label: int, exp_val: float, sequence: str) -> None:
        """Insert *sequence* into the top-k heap if its combined score is high enough.

        Score = label + exp_val.
        """
        if sequence in self.existing_seqs:
            return

        score = float(label) + exp_val
        item = (score, label, exp_val, sequence)
        if len(self.top_heap) < self.max_keep:
            heapq.heappush(self.top_heap, item)
            self.existing_seqs.add(sequence)
        else:
            min_score, _, _, _ = self.top_heap[0]
            if score > min_score:
                removed = heapq.heappop(self.top_heap)
                self.existing_seqs.remove(removed[3])
                heapq.heappush(self.top_heap, item)
                self.existing_seqs.add(sequence)

    def _save_top_heap(self, index: int) -> None:
        """Persist the current top-k heap to disk.

        Format:  Label: {label_int} | Expectation: {exp_val} | Seq: {seq}
        Score = Label + Expectation.
        """
        items = heapq.nlargest(
            len(self.top_heap), self.top_heap,
            key=lambda x: (x[0], x[1]),
        )
        path = f"./outputs/selfplay_top_heap/selfplay_top_heap{index}.txt"
        with open(path, "w") as f:
            for score, label, exp_val, seq in items:
                f.write(
                    f"Label: {label} | Expectation: {exp_val:.4f} | Seq: {seq}\n"
                )
