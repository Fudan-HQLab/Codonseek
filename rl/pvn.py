"""Policy-Value Network (PVN) — AlphaZero-style dual-head CNN.

Architecture
------------
Input  : (batch, 4, seq_len) — one-hot codon matrix from Sequencematrix.
Shared : Conv1d → BatchNorm → ReLU → 7× ResBlock (Conv1d).
Policy head : Conv1d(16) → BN → ReLU → FC(16·seq_len → 216) → log_softmax.
Value head  : Conv1d(8)  → BN → ReLU → FC(8·seq_len → 256) → ReLU → FC(1) → tanh.
"""

import math
import os
import time

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.config import CONFIG
from rl.mapper import CodonMapper, Sequencematrix
from rl._utils import timer


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Conv1d residual block with two 3×1 convolutions."""

    def __init__(self, num_filters: int = 256):
        super().__init__()
        self.conv1 = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(residual + x)


# ---------------------------------------------------------------------------
# Dual-head CNN
# ---------------------------------------------------------------------------

class PVN(nn.Module):
    """AlphaZero-style dual-head network for codon-sequence evaluation.

    Parameters
    ----------
    num_channels :
        Number of filters in convolutional layers.
    num_res_blocks :
        Number of residual blocks in the shared trunk.
    """

    def __init__(
        self,
        num_channels: int = 256,
        num_res_blocks: int = 7,
    ):
        super().__init__()
        seq_len = len(CONFIG["Protein_sequence"]) * 3

        # ---- Shared trunk ----
        self.conv_in = nn.Conv1d(4, num_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm1d(num_channels)
        self.act_in = nn.ReLU()

        self.res_blocks = nn.ModuleList(
            ResBlock(num_channels) for _ in range(num_res_blocks)
        )

        # ---- Policy head ----
        self.policy_conv = nn.Conv1d(num_channels, 16, kernel_size=1)
        self.policy_bn = nn.BatchNorm1d(16)
        self.policy_act = nn.ReLU()
        self.policy_fc = nn.Linear(16 * seq_len, 216)

        # ---- Value head ----
        self.value_conv = nn.Conv1d(num_channels, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm1d(8)
        self.value_act1 = nn.ReLU()
        self.value_fc1 = nn.Linear(8 * seq_len, 256)
        self.value_act2 = nn.ReLU()
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(log_softmax(policy), tanh(value))``."""
        # Shared trunk
        x = self.act_in(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        p = self.policy_act(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)
        p = F.log_softmax(self.policy_fc(p), dim=1)

        # Value head
        v = self.value_act1(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = self.value_act2(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


# ---------------------------------------------------------------------------
# PolicyValueNet — training & inference wrapper
# ---------------------------------------------------------------------------

class PolicyValueNet:
    """Wrapper around ``PVN`` that handles training, checkpointing, and the
    ``policy_value_fn`` interface consumed by MCTS.

    Parameters
    ----------
    model_file :
        Path to a ``.pkl`` state-dict (optional).
    use_gpu :
        If True, pin the model to the default CUDA device.
    """

    def __init__(
        self,
        model_file: str | None = None,
        use_gpu: bool = True,
    ):
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.net = PVN().to(self.device)

        self.l2_const = 2e-3
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.l2_const,
        )

        if model_file and os.path.exists(model_file):
            self.net.load_state_dict(
                torch.load(model_file, map_location=self.device, weights_only=True)
            )

        # Loss logging
        self.loss_file = None
        os.makedirs("./outputs/loss", exist_ok=True)
        try:
            self.loss_file = open("RL_loss.txt", "a", buffering=1)
        except OSError:
            pass

        self.train_losses: list[float] = []
        self.ploss: list[float] = []
        self.vloss: list[float] = []
        self.entropies: list[float] = []

        # Live-loss figure (non-blocking Agg backend)
        self.fig, self.ax = plt.subplots()

    # ------------------------------------------------------------------
    # Inference (used by MCTS)
    # ------------------------------------------------------------------

    def policy_value(self, state_batch: np.ndarray):
        """Evaluate a batch of encoded states.

        Parameters
        ----------
        state_batch :
            ``ndarray`` of shape ``(batch, 4, seq_len)``.

        Returns
        -------
        log_act_probs : ndarray (batch, 216)
        values : ndarray (batch, 1)
        """
        self.net.eval()
        state_t = torch.as_tensor(state_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            log_probs, values = self.net(state_t)
        return log_probs.cpu().numpy(), values.cpu().numpy()

    def policy_value_fn(self, codon_sequence: str):
        """MCTS-facing evaluation of a single codon-sequence prefix.

        Returns ``(seek_probs, value)`` where *seek_probs* is an iterable
        of ``(candidate_sequence, probability)`` pairs covering legal moves
        only, and *value* is a ``float``.
        """
        self.net.eval()

        legal_moves, candidate_seqs = CodonMapper(codon_sequence).available_codon(
            codon_sequence
        )
        legal_mask = legal_moves.astype(bool)

        state = Sequencematrix(codon_sequence)
        state_t = torch.as_tensor(
            np.ascontiguousarray(state.astype(np.float32)),
            device=self.device,
        )

        with torch.no_grad():
            log_probs, value = self.net(state_t)

        probs = np.exp(log_probs.cpu().numpy().astype(np.float32).flatten())
        probs *= legal_moves
        valid_probs = probs[legal_mask]

        return zip(candidate_seqs, valid_probs), value.item()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(
        self,
        state_batch: np.ndarray,
        mcts_probs: np.ndarray,
        winner_batch: np.ndarray,
        lr: float = 0.002,
    ):
        """Execute one training step and return ``(loss, policy_loss, value_loss, entropy)``."""
        self.net.train()

        state_t = torch.as_tensor(state_batch, dtype=torch.float32, device=self.device)
        target_p = torch.as_tensor(mcts_probs, dtype=torch.float32, device=self.device)
        target_v = torch.as_tensor(winner_batch, dtype=torch.float32, device=self.device)

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        self.optimizer.zero_grad()

        log_probs, values = self.net(state_t)

        value_loss = F.mse_loss(values.squeeze(-1), target_v)
        policy_loss = -(target_p * log_probs).sum(dim=1).mean()
        loss = value_loss + policy_loss

        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            entropy_val = -(torch.exp(log_probs) * log_probs).sum(dim=1).mean()
            self.train_losses.append(loss.item())
            self.ploss.append(policy_loss.item())
            self.vloss.append(value_loss.item())
            self.entropies.append(entropy_val.item())

        # Persist losses to disk
        if self.loss_file is not None:
            try:
                self.loss_file.write(
                    f"{loss.item():.4f}, {policy_loss.item():.4f}, "
                    f"{value_loss.item():.4f}, {entropy_val.item():.4f}\n"
                )
            except OSError:
                self.loss_file = None

        # Update live-loss chart
        self._update_loss_plot()

        return (
            loss.detach().cpu().numpy(),
            policy_loss.detach().cpu().numpy(),
            value_loss.detach().cpu().numpy(),
            entropy_val.detach().cpu().numpy(),
        )

    def _update_loss_plot(self) -> None:
        """Redraw the live training-loss chart and save to disk."""
        self.ax.clear()
        self.ax.plot(self.train_losses, color="#7FABD1", label="Training Loss")
        self.ax.plot(self.ploss, color="#8d4bbb", label="Policy Loss")
        self.ax.plot(self.vloss, color="#00a381", label="Value Loss")
        self.ax.plot(self.entropies, color="#F7AC53", label="Entropy")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training Loss")
        self.ax.legend()
        self.ax.grid(True)
        self.ax.set_xlim(0, len(self.train_losses))
        if self.train_losses:
            self.ax.set_ylim(0, math.ceil(max(self.train_losses)) + 1)

        os.makedirs("./outputs/loss", exist_ok=True)
        self.fig.savefig("./outputs/loss/training_loss1.png")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str) -> None:
        """Save the current PVN state-dict to *path*."""
        tmp_path = path + '.tmp'
        torch.save(self.net.state_dict(), tmp_path)
        os.replace(tmp_path, path)

    def __del__(self):
        if hasattr(self, "loss_file") and self.loss_file is not None and not self.loss_file.closed:
            self.loss_file.close()
        plt.close("all")
