# =============================================================================
# Source: Codon adaptation Language Model (CaLM)
# Reference: C. Outeiral and C. M. Deane, "Codon language embeddings provide
#   strong signals for use in protein engineering", Nature Machine Intelligence,
#   vol. 6, pp. 170179 (2024). doi: 10.1038/s42256-024-00791-0
# Repository: https://github.com/oxpig/CaLM
# License: MIT
#
# This file is adapted from the original CaLM pretrained module. It adds:
#   - load_calm(): module-level singleton cache (weights loaded once, kept on GPU)
#   - embed_sequences(): batched embedding (single forward pass for N sequences)
#   - configurable device parameter (removes hardcoded cuda:0)
# Included here as part of the codonseek project for inference-only usage.
# =============================================================================
import os
import pickle
from typing import Optional, Union, List

import torch
from .alphabet import Alphabet
from .sequence import CodonSequence
from .model import ProteinBertModel


# ---------------------------------------------------------------------------
# Default architecture hyper-parameters (CaLM publication)
# ---------------------------------------------------------------------------

_DEFAULT_ARGS = {
    "max_positions": 1024,
    "batch_size": 46,
    "accumulate_gradients": 40,
    "mask_proportion": 0.25,
    "leave_percent": 0.10,
    "mask_percent": 0.80,
    "warmup_steps": 1000,
    "weight_decay": 0.1,
    "lr_scheduler": "warmup_cosine",
    "learning_rate": 4e-4,
    "num_steps": 121000,
    "num_layers": 12,
    "embed_dim": 768,
    "attention_dropout": 0.0,
    "logit_bias": False,
    "rope_embedding": True,
    "ffn_embed_dim": 768 * 4,
    "attention_heads": 12,
}


class _ArgDict:
    """Simple namespace wrapper so attributes can be accessed via ``.name``."""
    def __init__(self, d):
        self.__dict__ = d


DEFAULT_ARGS = _ArgDict(_DEFAULT_ARGS)

# Folder where the checkpoint file lives (adjacent to this module)
_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "calm_weights")
_WEIGHTS_FILE = os.path.join(_WEIGHTS_DIR, "calm_weights.ckpt")

# ---------------------------------------------------------------------------
# Module-level cache: one model instance per weights path, per process.
# ---------------------------------------------------------------------------

_CALM_CACHE: dict = {}

def load_calm(
    weights_file: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> "CaLM":
    """Return a CaLM instance from a module-level cache.

    Subsequent calls with the same ``weights_file`` return the **same**
    instance without re-loading weights or re-allocating GPU memory.

    Parameters
    ----------
    weights_file :
        Path to the ``.ckpt`` (pickled state dict).  Defaults to
        ``calm/calm_weights/calm_weights.ckpt``.
    device :
        Target device.  Defaults to ``cuda:0`` if available else ``cpu``.

    Returns
    -------
    CaLM
    """
    key = weights_file or _WEIGHTS_FILE
    if key not in _CALM_CACHE:
        _CALM_CACHE[key] = CaLM(args=DEFAULT_ARGS, weights_file=weights_file, device=device)
    elif device is not None:
        # Ensure the cached model lives on the requested device
        _CALM_CACHE[key].to(device)
    return _CALM_CACHE[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class CaLM:
    """Codon adaptation Language Model.

    Wraps the 12-layer ProteinBERT encoder pre-trained on codon sequences.
    Provides convenience methods for extracting per-sequence embeddings.

    See Also
    --------
    :func:`load_calm`  recommended factory that caches the model instance.
    """

    def __init__(
        self,
        args: _ArgDict = DEFAULT_ARGS,
        weights_file: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        if weights_file is None:
            weights_file = _WEIGHTS_FILE
            if not os.path.exists(weights_file):
                print("Downloading model weights ")
                os.makedirs(_WEIGHTS_DIR, exist_ok=True)
                import requests
                url = "http://opig.stats.ox.ac.uk/data/downloads/calm_weights.pkl"
                with open(weights_file, "wb") as handle:
                    handle.write(requests.get(url).content)

        self.alphabet = Alphabet.from_architecture("CodonModel")
        self.model = ProteinBertModel(args, self.alphabet)
        self.bc = self.alphabet.get_batch_converter()

        with open(weights_file, "rb") as handle:
            state_dict = pickle.load(handle)
            self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)

    # -- convenience --------------------------------------------------------

    def to(self, device: torch.device) -> "CaLM":
        """Move the underlying model to *device* and update the default device."""
        self.model = self.model.to(device)
        self.device = device
        return self

    # -- embedding ----------------------------------------------------------

    def embed_sequence(
        self,
        sequence: Union[str, CodonSequence],
        average: bool = True,
    ) -> torch.Tensor:
        """Embed a single sequence.

        Parameters
        ----------
        sequence :
            DNA/codon string or ``CodonSequence`` instance.
        average :
            If True, return the mean-pooled representation over all codon
            positions  shape ``(1, embed_dim)``.
            If False, return the per-position tensor  ``(1, seq_len, embed_dim)``.

        Returns
        -------
        torch.Tensor
        """
        seq = sequence if isinstance(sequence, CodonSequence) else CodonSequence(sequence)
        tokens = self.tokenize(seq).to(self.device)
        repr_ = self.model(tokens, repr_layers=[12])["representations"][12]
        return repr_.mean(dim=1) if average else repr_

    def embed_sequences(
        self,
        sequences: List[Union[str, CodonSequence]],
        average: bool = True,
    ) -> torch.Tensor:
        """Embed a **batch** of sequences in a single forward pass.

        Unlike the original implementation (which loops over sequences one
        at a time), this method tokenises all sequences together and runs
        the model once, yielding substantial speed-ups on GPU.

        Parameters
        ----------
        sequences :
            List of DNA/codon strings or ``CodonSequence`` objects.
        average :
            If True, return mean-pooled representations  ``(batch, embed_dim)``.
            If False, return per-position tensors  ``(batch, seq_len, embed_dim)``.

        Returns
        -------
        torch.Tensor
        """
        if not sequences:
            raise ValueError("sequences must be non-empty")

        # Convert all inputs to (label, seq_str) pairs for BatchConverter
        batch_data = []
        for seq in sequences:
            if not isinstance(seq, CodonSequence):
                seq = CodonSequence(seq)
            batch_data.append(("", seq.seq))

        # Single batched tokenization
        _, _, tokens = self.bc(batch_data)
        tokens = tokens.to(self.device)

        # Single forward pass for the entire batch
        repr_ = self.model(tokens, repr_layers=[12])["representations"][12]

        if average:
            return repr_.mean(dim=1)  # (batch, embed_dim)
        return repr_  # (batch, seq_len, embed_dim)

    # -- tokenization -------------------------------------------------------

    def tokenize(self, seq: CodonSequence) -> torch.Tensor:
        """Return a ``(1, seq_len)`` tensor of token indices for *seq*."""
        assert isinstance(seq, CodonSequence), "seq must be CodonSequence"
        _, _, tokens = self.bc([("", seq.seq)])
        return tokens

    # -- raw forward --------------------------------------------------------

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full model (including LM head) on pre-tokenized input.

        Parameters
        ----------
        x :
            LongTensor of shape ``(batch, seq_len)`` containing token indices.

        Returns
        -------
        logits : torch.Tensor
        """
        x = x.to(self.device)
        return self.model(x)  # returns logits dict


