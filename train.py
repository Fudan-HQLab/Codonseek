"""Training loop for the Policy-Value Network.

Loads self-play data from per-collect pickle buffers, merges them,
and runs policy-gradient updates with KL-constrained early stopping.
"""

import math
import os
import pickle
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rl.config import CONFIG
from rl.mapper import Sequencematrix
from rl.pvn import PolicyValueNet


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_merge_data_buffers(
    buffer_dir: str = "./outputs/train_data_buffer",
    num_collectors: int = 3,
    pretrain_file: str | None = None,
    output_file: str = "biggest_train_data_buffer.pkl",
):
    """Merge data buffers from all collect processes into one training buffer.

    Parameters
    ----------
    buffer_dir : Directory containing `train_data_buffer_{1..N}.pkl`.
    num_collectors : Number of collect processes.
    pretrain_file : Optional path to a pretraining buffer.
    output_file : Where to write the merged buffer (also returned).

    Returns
    -------
    deque  of (sequence, mcts_probs, score) tuples.
    """
    merged_data = []
    total_samples = 0
    total_iters = 0

    # --- optional pretrain buffer ---
    if pretrain_file and os.path.exists(pretrain_file):
        try:
            with open(pretrain_file, "rb") as f:
                data = pickle.load(f)
            merged_data.extend(data["data_buffer"])
            total_samples += len(data["data_buffer"])
            total_iters += data.get("iters", 0)
            print(f"Loaded pretrain buffer: {len(data['data_buffer'])} samples")
        except Exception as e:
            print(f"Loading pretrain_file failed: {e}")

    # --- per-collect buffers ---
    for idx in range(1, num_collectors + 1):
        path = os.path.join(buffer_dir, f"train_data_buffer_{idx}.pkl")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            merged_data.extend(data["data_buffer"])
            total_samples += len(data["data_buffer"])
            total_iters += data.get("iters", 0)
        except Exception as e:
            print(f"Loading {path} failed: {e}")

    print(f"Total samples after merge: {total_samples}")
    random.shuffle(merged_data)

    biggest_buffer = {
        "data_buffer": merged_data,
        "total_samples": total_samples,
        "iters": total_iters,
    }
    with open(output_file, "wb") as f:
        pickle.dump(biggest_buffer, f)

    return merged_data


# ---------------------------------------------------------------------------
# TrainPipeline
# ---------------------------------------------------------------------------

class TrainPipeline:
    """Policy-value network training orchestrator.

    Periodically merges self-play data buffers, samples mini-batches,
    runs policy-gradient updates with KL early-stopping, and saves
    checkpoints.
    """

    def __init__(self, init_model: str | None = None):
        self.learn_rate = CONFIG["learning_rate"]
        self.lr_multiplier = 1.0
        self.batch_size = CONFIG["batch_size"]
        self.epochs = CONFIG["epochs"]
        self.kl_targ = CONFIG["kl_targ"]
        self.game_batch_num = CONFIG["game_batch_num"]
        self.seq = CONFIG["Protein_sequence"]

        self.policy_value_net: PolicyValueNet | None = None
        self.data_buffer = []
        self.iters = 0

        if init_model and os.path.exists(init_model):
            try:
                self.policy_value_net = PolicyValueNet(model_file=init_model)
                print(f"Loaded model from {init_model}")
            except Exception as e:
                print(f"Model load failed: {e} — initialising new network")
                self._init_new_network()
        else:
            self._init_new_network()

    def _init_new_network(self):
        print("Initialising new PolicyValueNet …")
        self.policy_value_net = PolicyValueNet()

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------

    def policy_update(self):
        """Run one policy-gradient update on a sampled mini-batch.

        Returns (total_loss, policy_loss, value_loss, entropy).
        """
        mini_batch = random.sample(self.data_buffer, self.batch_size)

        sequences = [d[0] for d in mini_batch]
        mcts_probs = np.array([d[1] for d in mini_batch], dtype=np.float64)
        scores = np.array([d[2] for d in mini_batch], dtype=np.float64)

        # Encode sequences as CNN-ready matrices
        state_batch = []
        for seq in sequences:
            mat = Sequencematrix(seq)
            state_batch.append(np.squeeze(mat, axis=0))

        # Old-policy probabilities (for KL monitoring)
        old_probs, _ = self.policy_value_net.policy_value(state_batch)

        for _ in range(self.epochs):
            loss, ploss, vloss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs,
                scores,
                self.learn_rate * self.lr_multiplier,
            )

            # KL divergence monitoring
            new_probs, _ = self.policy_value_net.policy_value(state_batch)
            kl = float(np.mean(
                np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1)
            ))

            if kl > self.kl_targ * 4:
                break

        # Adjust learning-rate multiplier based on KL
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        print(
            f"kl:{kl:.5f}  lr_multiplier:{self.lr_multiplier:.3f}  "
            f"loss:{loss}  entropy:{entropy}"
        )
        plt.close("all")
        return loss, ploss, vloss, entropy

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Main training loop — sleeps between updates, interrupted by Ctrl-C."""
        train_losses = []
        ploss = []
        vloss = []
        ce = []

        fig, ax = plt.subplots()

        try:
            for _ in range(self.game_batch_num):
                time.sleep(CONFIG["train_update_interval"])

                # Wait for enough data from collect processes
                while True:
                    self.data_buffer = load_and_merge_data_buffers(
                        num_collectors=CONFIG["num_runs"],
                        pretrain_file=CONFIG.get("pretrain_file"),
                        output_file=CONFIG["train_data_buffer_path"],
                    )
                    if self.data_buffer:
                        break
                    print("No data yet — retrying in 5 s …")
                    time.sleep(5)

                n_games = len(self.data_buffer) / (len(self.seq) / 3)
                print(f"len(self.data_buffer): {len(self.data_buffer)}")
                print(f"games: {n_games:.1f}")

                if n_games < CONFIG["collect_data_buffer"]:
                    self.policy_value_net.save_model(CONFIG["pytorch_model_path"])
                    continue

                loss, pl, vl, ent = self.policy_update()

                train_losses.append(loss)
                ploss.append(pl)
                vloss.append(vl)
                ce.append(ent)

                # --- live-loss chart ---
                ax.clear()
                ax.plot(train_losses, color="#7FABD1", label="Training Loss")
                ax.plot(ploss, color="#8d4bbb", label="Policy Loss")
                ax.plot(vloss, color="#00a381", label="Value Loss")
                ax.plot(ce, color="#F7AC53", label="Entropy")
                ax.set_xlabel("Training steps")
                ax.set_ylabel("Loss")
                ax.set_title("Training Loss (LBA)")
                ax.legend()
                ax.grid(True)
                ax.set_xlim(0, len(train_losses))
                if train_losses:
                    ax.set_ylim(0, math.ceil(max(train_losses)) + 1)

                os.makedirs("./outputs/loss", exist_ok=True)
                fig.savefig("./outputs/loss/training_loss.png")

                # Save checkpoint
                self.policy_value_net.save_model(CONFIG["pytorch_model_path"])

        except KeyboardInterrupt:
            print("Training interrupted.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    pipeline = TrainPipeline(init_model="current_policy.pkl")
    pipeline.run()


if __name__ == "__main__":
    main()