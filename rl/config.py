"""
Configuration parameters for the codonseek project.

All tunable hyper-parameters and file paths live here.
Genetic-code lookup tables have been moved to ``codon_mapper.py``.
"""

CONFIG = {
    # --- Protein target ---
    "Protein_sequence": "MSSQIRQNYSTDVEAAVNSLVNLYLQASYTYLSLGFYFDRDDVALEGVSHFFRELAEEKREGYERLLKMQNQRGGRALFQDIKKPAEDEWGKTPDAMKAAMALEKKLNQALLDLHALGSARTDPHLCDFLETHFLDEEVKLIKKMGDHLTNLHRLGGPEAGLGEYLFERLTLKHD*",

    # --- MCTS ---
    "dirichlet": 0.2,       # Dirichlet noise weight for exploration
    "play_out": 220,        # Number of MCTS simulations per move
    "c_puct": 10,           # Exploration constant for UCT

    # --- Self-play ---
    "buffer_size": 1_000_000,
    "collect_data_buffer": 2000,  # minimum games before a training update
    "num_runs": 3,                # number of parallel collect processes
    "train_update_interval": 5, # seconds between training updates

    # --- Training ---
    "learning_rate": 1e-7,
    "batch_size": 128,
    "kl_targ": 0.02,
    "epochs": 5,
    "game_batch_num": 500,
    "use_frame": "pytorch",

    # --- Model paths ---
    "pytorch_model_path": "current_policy.pkl",
    "train_data_buffer_path": "biggest_train_data_buffer.pkl",
    "ten_model_path": "ten-weights",
    "pretrain_file": "pretrain_start/pretrain.pkl",  # warm-start data

    # --- GPU ---
    "gpu_devices": "0",  # GPU indices for collect processes, e.g. "0" or "0,1"
    "use_redis": False,
}
