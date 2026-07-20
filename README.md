<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.13.2-3776AB?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/RL-MCTS%20%2B%20AlphaZero-6DB33F" alt="RL">
</p>

# Codonseek: Reinforcement Learning-Based Codon Optimization Guided by Experimentally Grounded Reward Signals
Junyuan Zeng, Hanyao Jiang, Gaoxing Guo, Jingqi Wang, Xinzhou Qian, Enze Zhang, Chunlong Wen, Jungang Zhou, Hong Lu, Qiang Huang, and Yao Yu

codonseek is a codon optimisation framework powered by AlphaZero-style
reinforcement learning. Given a target protein -- human ferritin light chain
(FTL) -- it searches the space of synonymous codon sequences for variants
that maximise translation efficiency (TE).

## How It Works

```
+----------+     +--------------+     +-------------+
| CaLM     |---->| TEN (Reward) |---->| MCTS + PVN  |
| Embedding|     | Score Model  |     | Self-Play   |
+----------+     +--------------+     +-------------+
```

1. **CaLM** (Codon adaptation Language Model) -- a pretrained Transformer that
   encodes codon sequences into 768-dimensional embeddings.
   [Outeiral & Deane, *Nat. Mach. Intell.* 2024]
2. **TEN** (Translation Efficiency Network) -- a reward model that maps CaLM
   embeddings to a 5-class translation efficiency score.
3. **MCTS + PVN** -- Monte Carlo Tree Search paired with a policy-value
   dual-head CNN, iteratively discovering high-scoring sequences through
   self-play.

## Project Structure

```
codonseek/
+-- calm/                       # CaLM pretrained model (inference-only)
|   +-- model.py                #   Transformer architecture
|   +-- modules.py              #   Layers & activation functions
|   +-- multihead_attention.py  #   Multi-head self-attention
|   +-- pretrained.py           #   Model loader & tokenizer
|   +-- alphabet.py             #   Codon alphabet (64 triplets)
|   +-- calm_weights/           #   Pretrained weights (~327 MB)
+-- ten/                        # Translation Efficiency Network
|   +-- model.py                #   MLP + Transformer residual blocks
|   +-- train_ten.py            #   5-fold cross-validation training
|   +-- embed_fasta.py          #   Batch CaLM embedding generation
+-- rl/                         # Reinforcement learning core
|   +-- config.py               #   Hyperparameters & paths
|   +-- mapper.py               #   Codon encoding matrix & legal-move generator
|   +-- pvn.py                  #   Policy-Value dual-head CNN (AlphaZero-style)
|   +-- mcts.py                 #   Monte Carlo Tree Search & player
|   +-- game.py                 #   Self-play engine (SelfSampling)
|   +-- collect.py              #   Multi-process data-collection pipeline
|   +-- _utils.py               #   Timer decorator & shared helpers
+-- pretrain_start/             # Pretraining data generation
|   +-- generate_traindata.py   #   Synonymous-variant generation + TEN scoring -> FASTA
|   +-- generate_pretrainpkl.py #   FASTA -> MCTS training buffer pickle
+-- analysis/                   # Post-run analysis & visualisation
|   +-- analyze.py              #   Full pipeline (postprocess + figures + archive)
|   +-- postprocess/            #   Merge, deduplicate, rank, CSV export
|   +-- figures/                #   Cluster, distribution, heatmap, stacked, PVN-loss plots
+-- outputs/                    # Runtime output (re-created each run)
|   +-- selfplay_top_heap/      #   Best sequences per self-play episode
|   +-- mcts_top_heap/          #   Best paths from MCTS search
|   +-- RLscore/                #   RL score logs
|   +-- loss/                   #   Training loss curves
|   +-- train_data_buffer/      #   Experience replay buffer
+-- run.py                      # Launch multi-process self-play data collection
+-- train.py                    # PVN training loop (load -> policy gradient -> KL early-stop)
+-- ten_weights.pth             # Pretrained TEN weights
```

## Getting Started

### Running the Pipeline

#### 1. Generate Pretraining Data (optional -- `pretrain.pkl` already provided)

```bash
cd codonseek
python pretrain_start/generate_traindata.py       # produce scored FASTA
python pretrain_start/generate_pretrainpkl.py \
    hFTLcompany20%.fasta pretrain_start/pretrain.pkl
```

#### 2. Launch Self-Play & Training

Open two terminals:

```bash
# Terminal 1 -- data collection
python run.py

# Terminal 2 -- policy-value network training
python train.py
```

The two processes coordinate through a shared checkpoint file
(`current_policy.pkl`) and the data buffer directory
(`outputs/train_data_buffer/`).

#### 3. Analyse Results

```bash
python analysis/analyze.py              # full pipeline
python analysis/analyze.py --dry-run    # preview without executing
```

## Core Configuration

All key parameters live in [rl/config.py](rl/config.py):

| Parameter          | Default           | Description                                   |
|--------------------|-------------------|-----------------------------------------------|
| `Protein_sequence` | aa seq            | Target protein amino-acid sequence            |
| `play_out`         | 220               | MCTS simulations per move                     |
| `c_puct`           | 10                | UCT exploration constant                      |
| `learning_rate`    | 1e-7              | Policy network learning rate                  |
| `kl_targ`          | 0.02              | KL-divergence early-stopping threshold        |
| `num_runs`         | 3                 | Number of parallel data-collection processes  |
| `buffer_size`      | 1,000,000         | Training replay buffer capacity               |

## References

- **CaLM** -- C. Outeiral & C. M. Deane, "Codon language embeddings provide
  strong signals for use in protein engineering", *Nature Machine Intelligence*
  **6**, 170-179 (2024).  [GitHub: oxpig/CaLM](https://github.com/oxpig/CaLM)
- **AlphaZero** -- D. Silver et al., "A general reinforcement learning
  algorithm that masters chess, shogi, and Go through self-play",
  *Science* **362**, 1140-1144 (2018).

## License

The CaLM submodule is adapted from [oxpig/CaLM](https://github.com/oxpig/CaLM)
under the MIT License. All other code in this repository is original.
