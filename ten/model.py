"""
Translation Efficiency Network (TEN)

Multi-layer perceptron with transformer residual blocks.
Takes a 768-dim CaLM embedding and predicts a 5-class translation
efficiency score.

Reference: built for the codonseek project as a reward model in
reinforcement-learning-based codon optimisation.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Architecture constants
# ---------------------------------------------------------------------------

INPUT_SIZE: int = 768
HIDDEN_SIZES: List[int] = [8, 8, 4, 4, 2]
NUM_CLASSES: int = 5
NHEAD: int = 8
DROPOUT: float = 0.25

# ---------------------------------------------------------------------------
# Score mapping
# ---------------------------------------------------------------------------

# Weighted-sum coefficients for each of the five classes.
# Class 0 (lowest TE) ... Class 4 (highest TE).
LABEL_WEIGHTS: List[float] = [-1.0, -0.5, -0.1, 0.5, 1.0]

# ---------------------------------------------------------------------------
# Transformer residual block
# ---------------------------------------------------------------------------


class AttentionResidualBlock(nn.Module):
    """Transformer encoder layer applied per-sample (adds a virtual batch dim
    internally so the layer can process a single sequence).

    *nhead* is automatically clamped to a divisor of *embed_dim*
    (required by `nn.TransformerEncoderLayer`).
    """

    def __init__(
        self,
        embed_dim: int,
        nhead: int = NHEAD,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        _nhead = min(nhead, embed_dim)
        while embed_dim % _nhead != 0:
            _nhead -= 1
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=_nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TransformerEncoderLayer expects (S, N, E); add a seq-len dim
        # at position 0 and remove it afterwards.
        if x.dim() == 2:
            x = x.unsqueeze(0)          # (B, E) → (1, B, E)
        out = self.transformer_layer(x)
        out = out.squeeze(0)            # (1, B, E) → (B, E)
        return out


# ---------------------------------------------------------------------------
# Network definition
# ---------------------------------------------------------------------------


class TEN(nn.Module):
    """Translation Efficiency Network.

    Architecture
    ------------
    Input (batch, 768)
      Linear(768, 8) + ReLU + BatchNorm + Dropout
      AttentionResidualBlock(8)
      AttentionResidualBlock(8)
      AttentionResidualBlock(8)
      Linear(8, 2) + ReLU + BatchNorm + Dropout + Linear(2, 5)
      logits (batch, 5)

    Note: the three `AttentionResidualBlock` layers all operate at
    `embed_dim = hidden_sizes[0]` (= 8) because a standard Transformer
    encoder layer preserves the input dimensionality.  The intermediate
    `hidden_sizes[1:-1]` values are only used to determine *how many*
    residual blocks to create, not their width.
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_sizes: Optional[List[int]] = None,
        num_classes: int = NUM_CLASSES,
        nhead: int = NHEAD,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = HIDDEN_SIZES

        # Feature trunk
        layers: List[nn.Module] = [
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Dropout(dropout),
        ]
        # All residual blocks operate at hidden_sizes[0] (transformer layers
        # preserve dimension; hidden_sizes[1:-1] only controls the count).
        prev_size = hidden_sizes[0]
        for _ in hidden_sizes[1:-1]:
            layers.append(
                AttentionResidualBlock(embed_dim=prev_size, nhead=nhead)
            )
        self.layers = nn.Sequential(*layers)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[-1]),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[-1], num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=1.0)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape (batch, num_classes)."""
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Reward-model wrapper
# ---------------------------------------------------------------------------


class TENReward:
    """Lightweight reward model wrapping a trained TEN checkpoint.

    Usage::

        reward = TENReward("checkpoint.pth", device="cuda:0")
        score = reward.score(embedding)            # single sample  (768,)
        scores = reward.score_batch(embeddings)    # (N, 768)  (N,)

    The returned score is `argmax_class + weighted_sum(probs)`.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda:0"):
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model = TEN(
            input_size=ckpt["input_size"],
            hidden_sizes=ckpt["hidden_sizes"],
            num_classes=ckpt["num_classes"],
            nhead=ckpt.get("nhead", NHEAD),
        )
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        self._weights = torch.tensor(LABEL_WEIGHTS, device=self.device)

    @torch.no_grad()
    def _predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run TEN and return (softmax_probs, argmax_labels)."""
        logits = self.model(x.to(self.device))
        probs = torch.softmax(logits, dim=-1)
        labels = torch.argmax(probs, dim=-1)
        return probs, labels

    def score(
        self,
        embedding: torch.Tensor,
    ) -> Tuple[float, float, int]:
        """Evaluate a single embedding.

        Parameters
        ----------
        embedding :
            Tensor of shape `(768,)` or `(1, 768)`.

        Returns
        -------
        score : float
            label + weighted expectation.
        expectation : float
            weighted sum of class probabilities.
        label : int
            argmax class index (0 … 4).
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        probs, labels = self._predict(embedding)
        exp = (probs * self._weights).sum(dim=-1).item()
        lab = labels.item()
        return lab + exp, exp, lab

    def score_batch(
        self,
        embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate a batch of embeddings.

        Parameters
        ----------
        embeddings :
            Tensor of shape `(batch, 768)`.

        Returns
        -------
        scores : (batch,)
            label + weighted expectation.
        expectations : (batch,)
            weighted sum of class probabilities.
        labels : (batch,)
            argmax class indices.
        """
        probs, labels = self._predict(embeddings)
        exp = (probs * self._weights).sum(dim=-1)
        return labels.float() + exp, exp, labels