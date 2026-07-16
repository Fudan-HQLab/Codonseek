"""Translation Efficiency Network (TEN).

Modules
-------
model       : TEN architecture and TENReward wrapper
train_ten   : 5-fold cross-validation training script
embed_fasta : batch-generate CaLM embeddings from a codon FASTA
"""
from .model import TEN, TENReward

__all__ = ["TEN", "TENReward"]
