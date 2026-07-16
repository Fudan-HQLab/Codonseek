"""Reinforcement learning pipeline for codon optimisation.

Modules
-------
config    : Hyper-parameters and paths
game      : SelfSampling — self-play game engine
mcts      : MCTS tree search and player
pvn       : PolicyValueNet — AlphaZero dual-head CNN
mapper    : Codon sequence encoding and legal-move generation
collect   : Data-collection pipeline
_utils    : Shared helpers (timer decorator, etc.)
"""
