# =============================================================================
# Source: Codon adaptation Language Model (CaLM)
# Reference: C. Outeiral and C. M. Deane, "Codon language embeddings provide
#   strong signals for use in protein engineering", Nature Machine Intelligence,
#   vol. 6, pp. 170179 (2024). doi: 10.1038/s42256-024-00791-0
# Repository: https://github.com/oxpig/CaLM
# License: MIT
#
# Public API for inference-only usage in the codonseek project.
# =============================================================================

from .pretrained import CaLM, load_calm

__all__ = ["CaLM", "load_calm"]

