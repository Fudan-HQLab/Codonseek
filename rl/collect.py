"""Self-play data-collection pipeline.

Each collect process runs an independent self-play loop, writes
game data to per-process pickle buffers, and periodically reloads
the latest policy network when an updated checkpoint is detected.
"""

import logging
import os
import pickle
import time
import traceback
from collections import deque
import torch

from rl.config import CONFIG
from rl.game import SelfSampling
from rl.mcts import MCTSPlayer
from rl.pvn import PolicyValueNet

# ---------------------------------------------------------------------------
# Logging is configured *after* multiprocessing forks so each child gets
# its own stream handler keyed by COLLECT_INDEX.  The _setup_logging()
# helper is called inside main() and CollectPipeline.__init__.
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    idx = os.environ.get("COLLECT_INDEX", "0")
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [%(levelname)-7s] collect_{idx} | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"collect_{idx}.log"),
        ],
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CollectPipeline
# ---------------------------------------------------------------------------


class CollectPipeline:
    """Runs self-play games, persists training data, and reloads the
    policy network when an updated checkpoint appears on disk."""

    def __init__(self, index: int = 0):
        _setup_logging()
        self.idx = self._resolve_index(index)

        # Hyper-parameters
        self.n_playout = CONFIG["play_out"]
        self.c_puct = CONFIG["c_puct"]

        # Data buffer
        self.buffer_size = CONFIG["buffer_size"]
        self.data_buffer: deque = deque(maxlen=self.buffer_size)
        self.iters = 0

        # Paths
        self.pytorch_model_path = CONFIG["pytorch_model_path"]
        self.ten_model_path = CONFIG["ten_model_path"]

        # Model freshness tracking
        self.model_version = None

        # Self-play engine (owns CaLM + TENReward internally)
        self.self_sampling = SelfSampling()

        # Ensure output directories exist
        os.makedirs("./outputs/selfplay_time", exist_ok=True)
        os.makedirs("./outputs/train_data_buffer", exist_ok=True)

        # Initial model load
        self._load_policy_net()
        self._build_mcts_player()

    # ------------------------------------------------------------------
    # Index resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_index(index: int) -> int:
        """Resolve the collect-process index (environment overrides argument)."""
        env_index = os.environ.get("COLLECT_INDEX")
        if env_index is not None:
            try:
                return int(env_index)
            except (ValueError, TypeError):
                pass
        return index

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def _load_policy_net(self) -> None:
        """Load (or create) the policy-value network."""
        if os.path.exists(self.pytorch_model_path):
            self.policy_value_net = PolicyValueNet(
                model_file=self.pytorch_model_path,
                use_gpu=True,
            )
            self.model_version = os.path.getmtime(self.pytorch_model_path)
        else:
            logger.warning(
                "No policy checkpoint at %s; initialising from scratch.",
                self.pytorch_model_path,
            )
            self.policy_value_net = PolicyValueNet(use_gpu=True)
            self.model_version = None

    def _build_mcts_player(self) -> None:
        """Construct a fresh MCTS player backed by the current policy network.

        Explicitly deletes the previous `mcts_player` before creating a new
        one so that stale GPU tensors (especially the old TENReward inside
        MCTS) are freed before the new one allocates — this avoids silent OOM
        crashes when reloading checkpoints mid-run.
        """
        if hasattr(self, "mcts_player"):
            del self.mcts_player
            torch.cuda.empty_cache()

        self.mcts_player = MCTSPlayer(
            policy_value_function=self.policy_value_net.policy_value_fn,
            ten_model_path=self.ten_model_path,
            collect_index=self.idx,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=1,
        )

    def _refresh_model_if_stale(self) -> None:
        """Reload the policy network when a newer checkpoint is detected."""
        if not os.path.exists(self.pytorch_model_path):
            return
        mtime = os.path.getmtime(self.pytorch_model_path)
        if mtime == self.model_version:
            return
        logger.info("New policy checkpoint (mtime=%s); reloading ...", mtime)
        self._load_policy_net()
        self._build_mcts_player()

    # ------------------------------------------------------------------
    # Self-play
    # ------------------------------------------------------------------

    def collect_selfplay_data(self, n_games: int = 1) -> int:
        """Play *n_games* self-play games and persist the resulting buffer."""
        for i in range(n_games):
            self._refresh_model_if_stale()

            t0 = time.perf_counter()
            play_data = self.self_sampling.self_play("", self.mcts_player, self.idx)
            elapsed = time.perf_counter() - t0
            logger.info(
                "Collect #%d, game %d -- %.2f s", self.idx, i + 1, elapsed
            )

            with open(
                f"./outputs/selfplay_time/selfplay_time_{self.idx}.txt", "a"
            ) as f:
                f.write(f"game:{i + 1}  time:{elapsed:.4f} s\n")

            play_data = list(play_data)
            self._persist_buffer(play_data)

        return self.iters

    def _persist_buffer(self, play_data: list) -> None:
        """Append *play_data* to the per-process buffer with atomic writes."""
        buffer_file = (
            f"./outputs/train_data_buffer/train_data_buffer_{self.idx}.pkl"
        )

        if os.path.exists(buffer_file):
            for attempt in range(20):
                try:
                    with open(buffer_file, "rb") as fh:
                        saved = pickle.load(fh)
                    self.data_buffer = saved["data_buffer"]
                    self.iters = saved["iters"]
                    break
                except (pickle.UnpicklingError, EOFError, OSError):
                    if attempt < 19:
                        time.sleep(3)
                    else:
                        logger.warning("Could not read buffer file, starting fresh.")

        self.iters += 1
        self.data_buffer.extend(play_data)

        data_dict = {"data_buffer": self.data_buffer, "iters": self.iters}
        tmp_path = buffer_file + ".tmp"
        with open(tmp_path, "wb") as fh:
            pickle.dump(data_dict, fh)
        os.replace(tmp_path, buffer_file)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Endless self-play loop -- interrupt with Ctrl-C."""
        logger.info(
            "Collect #%d started  (c_puct=%.2f)", self.idx, self.c_puct)
        k = 0
        try:
            while True:
                k += 1
                logger.info(
                    "Collect #%d -- round %d  c_puct=%.2f", self.idx, k, self.c_puct,
                )
                self.collect_selfplay_data()

                if k % 500 == 0 and self.c_puct > 10:
                    self.c_puct = max(self.c_puct * 0.9, 10)
                    logger.info("c_puct decayed to %.2f", self.c_puct)

        except KeyboardInterrupt:
            logger.info("Collect #%d interrupted.", self.idx)
        except Exception:
            logger.error(
                "Collect #%d crashed (round %d):\n%s",
                self.idx, k, traceback.format_exc(),
            )
            raise


# ---------------------------------------------------------------------------
# Entry point (invoked by multiprocessing in run.py)
# ---------------------------------------------------------------------------


def main() -> None:
    _setup_logging()
    index = int(os.environ.get("COLLECT_INDEX", "0"))
    logger.info("Collect #%d initialising.", index)
    pipeline = CollectPipeline(index=index)
    pipeline.run()


if __name__ == "__main__":
    main()
