"""Batch-generate CaLM embeddings for a codon-sequence FASTA file.

Reads a FASTA file of codon sequences, embeds each one using the
codon adaptation Language Model (CaLM), and writes the resulting
768-dim vectors to a text file in bracket-list format (one embedding
per line).

Usage
-----
    python 2codon_embedding768.py

Default paths are set in ``CONFIG`` at the bottom of this script.
"""

import argparse
import logging
from pathlib import Path

import torch
from Bio import SeqIO
from calm import load_calm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "fasta_path": "../data/ppa_codon_seq.fasta",
    "txt_path": "../data/ppa_embeddings.txt",
    "batch_size": 64,
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def embed_fasta(
    fasta_path: str,
    output_path: str,
    batch_size: int = 64,
    device: str = "cuda:0",
) -> None:
    """Embed all codon sequences in *fasta_path* using CaLM and write to *output_path*.

    Parameters
    ----------
    fasta_path :
        Path to a FASTA file where each record is a codon sequence.
    output_path :
        Destination text file: one line per sequence, each line a Python
        list-of-floats (bracket-delimited, comma-separated) of shape (768,).
    batch_size :
        Number of sequences to embed in a single CaLM forward pass.
    device :
        PyTorch device string (e.g. ``"cuda:0"`` or ``"cpu"``).
    """
    calm = load_calm(device=torch.device(device))

    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        logger.warning("No records found in %s", fasta_path)
        return

    logger.info("Embedding %d sequences (batch_size=%d) …", len(records), batch_size)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad(), open(output_path, "w") as f:
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            sequences = [str(r.seq) for r in batch]

            embeddings = calm.embed_sequences(sequences).cpu().numpy()

            for emb in embeddings:
                line = str([round(x, 6) for x in emb.tolist()])
                f.write(line + "\n")

            if (i + len(batch)) % 100 == 0 or i == 0:
                logger.info("  %d / %d", i + len(batch), len(records))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Generate CaLM embeddings for a codon-sequence FASTA."
    )
    parser.add_argument(
        "--fasta",
        default=CONFIG["fasta_path"],
        help="Input FASTA file (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=CONFIG["txt_path"],
        help="Output text file (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=CONFIG["batch_size"],
        help="Sequences per CaLM forward pass (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch device (default: cuda:0)",
    )
    args = parser.parse_args()

    embed_fasta(
        fasta_path=args.fasta,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()

