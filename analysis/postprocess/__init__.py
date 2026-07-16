"""
Post-processing utilities for MCTS self-play output files.

merge     : combine multiple text output files
dedup     : remove duplicate sequences from top-heap dumps
rank_top  : extract top-100 sequences by score/expectation
rank_score: filter and rank by specific score thresholds
deal_csv  : parse unstructured text output into CSV
"""
