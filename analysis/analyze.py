#!/usr/bin/env python
"""codonseek post-run analysis pipeline.

1. Post-process raw data inside outputs/ (combine, deduplicate, rank, csv).
2. Generate all figures from the processed data.
3. Archive everything (outputs/ contents, project-level logs, loss files,
   model checkpoint) into analysis/results/<timestamp>/.
4. Clean up runtime detritus (log files, merged buffer, etc.) and
   re-create empty output subdirectories so the next run can start clean.

Usage
-----
    cd codonseek && python analysis/analyze.py                 # full pipeline
    python analysis/analyze.py --skip-figures  # post-process only
    python analysis/analyze.py --dry-run       # show what would be done
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent                       # analysis/
PROJECT = HERE.parent                                        # codonseek/
OUTPUTS = PROJECT / "outputs"                                # raw runtime data
RESULTS_DIR = HERE / "results"                               # archived runs
POSTPROCESS_DIR = HERE / "postprocess"
FIGURES_DIR = HERE / "figures"

# Subdirectories that must exist under outputs/ for the RL pipeline to run.
OUTPUT_SUBDIRS = [
    "RLscore",
    "mcts_top_heap",
    "selfplay_top_heap",
    "selfplay_time",
    "train_data_buffer",
    "loss",
]

logger = logging.getLogger("analysis")

# Intermediate files created during post-processing (deleted after use).
INTERMEDIATE_FILES = [
    "combined_texts.txt",
    "undealtotal_top_heap.txt",
    "undealtotal_top_heap.csv",
    "all_one.txt",
    "temp_seq.db",
]

POSTPROCESS_STEPS = [
    ("combine",      POSTPROCESS_DIR / "combine.py"),
    ("deduplicate",  POSTPROCESS_DIR / "deduplicate.py"),
    ("rank_score",   POSTPROCESS_DIR / "rank_score.py"),
    ("rank_top",     POSTPROCESS_DIR / "rank_top.py"),
    ("deal_csv",     POSTPROCESS_DIR / "deal_csv.py"),
]

FIGURE_STEPS = [
    ("cluster",               FIGURES_DIR / "cluster.py"),
    ("distribution_paint",    FIGURES_DIR / "distribution_paint.py"),
    ("heap_map",              FIGURES_DIR / "heap_map.py"),
    ("pvnlossdraw",           FIGURES_DIR / "pvnlossdraw.py"),
    ("stacked_map",           FIGURES_DIR / "stacked_map.py"),
]

# Files at the project root that should be archived (logs, loss, checkpoint).
PROJECT_ARTEFACTS = [
    "collect_*.log",
    "RL_loss.txt",
    "current_policy.pkl",
]

# Files at the project root that should be DELETED (merged buffer).
CLEANUP_FILES = [
    "biggest_train_data_buffer.pkl",
    "collect_*.log",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_script(name: str, script: Path, cwd: Path) -> bool:
    logger.info("[%s] running ...", name)
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(cwd),
        capture_output=False,
    )
    ok = result.returncode == 0
    if ok:
        logger.info("[%s] OK", name)
    else:
        logger.error("[%s] FAILED (exit %d)", name, result.returncode)
    return ok


def _cleanup_intermediates(cwd: Path) -> None:
    for name in INTERMEDIATE_FILES:
        path = cwd / name
        if path.exists():
            path.unlink()
            logger.info("  removed %s", name)


def _archive_project_artefacts(run_dir: Path) -> None:
    """Copy logs, loss files, and checkpoint from the project root into
    run_dir/, then delete them from the root."""
    for pattern in PROJECT_ARTEFACTS:
        for path in PROJECT.glob(pattern):
            dest = run_dir / path.name
            shutil.move(str(path), str(dest))
            logger.info("  archived %s", path.name)


def _cleanup_project_root() -> None:
    """Delete transient files from the project root."""
    for pattern in CLEANUP_FILES:
        for path in PROJECT.glob(pattern):
            path.unlink()
            logger.info("  removed %s", path.name)


def _archive_outputs(run_dir: Path) -> None:
    """Move everything under outputs/ into run_dir/, then re-create the
    empty subdirectory skeleton for the next run."""
    if not OUTPUTS.exists():
        return

    for item in list(OUTPUTS.iterdir()):
        dest = run_dir / item.name
        shutil.move(str(item), str(dest))
        logger.info("  archived outputs/%s", item.name)

    for sub in OUTPUT_SUBDIRS:
        (OUTPUTS / sub).mkdir(parents=True, exist_ok=True)
    logger.info("outputs/ re-initialised for next run.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="codonseek analysis pipeline")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-postprocess", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    # ----- 1.  Set up results directory ------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_dir)

    # ----- 2.  Post-processing (inside outputs/) ---------------------------
    if not args.skip_postprocess:
        logger.info("Post-processing ...")
        for name, script in POSTPROCESS_STEPS:
            if args.dry_run:
                logger.info("  [DRY RUN] %s", name)
                continue
            _run_script(name, script, cwd=PROJECT)

        _cleanup_intermediates(OUTPUTS)
        logger.info("Post-processing complete.")
    else:
        logger.info("Skipping post-processing (--skip-postprocess).")

    # ----- 3.  Figures -----------------------------------------------------
    if not args.skip_figures:
        logger.info("Generating figures ...")
        for name, script in FIGURE_STEPS:
            if args.dry_run:
                logger.info("  [DRY RUN] %s", name)
                continue
            _run_script(name, script, cwd=PROJECT)
        logger.info("Figures complete.")
    else:
        logger.info("Skipping figures (--skip-figures).")

    # ----- 4.  Archive & clean ---------------------------------------------
    if not args.dry_run:
        # Archive project-level artefacts BEFORE moving outputs (so relative
        # paths inside log files remain meaningful).
        _archive_project_artefacts(run_dir)

        _archive_outputs(run_dir)

        _cleanup_project_root()

        summary = run_dir / "run_info.txt"
        with open(summary, "w") as f:
            f.write(f"codonseek analysis run - {timestamp}\n")
            f.write(f"Archived from: {OUTPUTS}\n")
        logger.info("Done - all data archived to %s", run_dir)
    else:
        logger.info("[DRY RUN] Would archive outputs/ to %s", run_dir)


if __name__ == "__main__":
    main()