"""Disk-level deduplication of top-heap sequences using SQLite.

Reads a combined text file of scored sequences and removes exact
duplicates (by MD5 of the sequence), writing unique records only.
This is a disk-backed alternative to in-memory dedup for large inputs.
"""

import hashlib
import os
import sqlite3


def sqlite_deduplicate(input_path: str, output_path: str, batch_size: int = 10000) -> None:
    """Deduplicate scored-sequence records using a SQLite-backed unique index.

    Each record is identified by the MD5 hash of its codon sequence
    (extracted from ``| Seq: ...``).  Duplicates are silently dropped
    via ``INSERT OR IGNORE``.

    Parameters
    ----------
    input_path : str
        Path to the combined top-heap text file.
    output_path : str
        Path where unique (first-seen) records are written.
    batch_size : int
        Number of rows to buffer before committing to the database.
    """
    DB_PATH = "temp_seq.db"

    # --- Initialise database -----------------------------------------------
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sequences (
            seq_md5  BLOB PRIMARY KEY,
            full_text TEXT
        )
    """)

    # --- Batch insert with MD5 dedup ---------------------------------------
    buffer: list = []
    with open(input_path, "r") as fin:
        for line in fin:
            try:
                seq = line.split("| Seq: ")[1].strip()
            except IndexError:
                continue

            seq_hash = hashlib.md5(seq.encode()).digest()
            buffer.append((seq_hash, line))

            if len(buffer) >= batch_size:
                cursor.executemany(
                    "INSERT OR IGNORE INTO sequences VALUES (?, ?)",
                    buffer,
                )
                conn.commit()
                buffer.clear()

        # Flush remaining rows
        if buffer:
            cursor.executemany(
                "INSERT OR IGNORE INTO sequences VALUES (?, ?)",
                buffer,
            )
            conn.commit()

    # --- Write unique results to output ------------------------------------
    cursor.execute("SELECT full_text FROM sequences ORDER BY rowid")
    with open(output_path, "w") as fout:
        for row in cursor:
            fout.write(row[0])

    # --- Cleanup -----------------------------------------------------------
    conn.close()
    os.remove(DB_PATH)


if __name__ == "__main__":
    input_file = "combined_texts.txt"
    output_file = "undealtotal_top_heap.txt"

    # Use the disk-backed method for memory-constrained environments.
    # For smaller datasets (< ~10^5 records) an in-memory set-based dedup
    # is also available (see git history).
    sqlite_deduplicate(input_file, output_file, batch_size=50000)
