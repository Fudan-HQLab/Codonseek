"""Convert the deduplicated top-heap text file into a CSV with
Score, Label, Expectation, Seq columns.

Reads from `undealtotal_top_heap.txt` (produced by deduplicate.py)
and writes `total_top_heap.csv`.
"""

import pandas as pd


INPUT_TXT = "./outputs/undealtotal_top_heap.txt"
OUTPUT_CSV = "./outputs/total_top_heap.csv"
CHUNK_SIZE = 100_000


def process_chunk(chunk):
    def split_row(row):
        parts = row.split("|")
        label_part = parts[0].strip()           # "Label: 3"
        exp_part = parts[1].strip()             # "Expectation: 0.1234"
        seq_part = parts[2].strip()             # "Seq: ATG..."

        label = int(label_part.replace("Label: ", ""))
        exp_val = float(exp_part.replace("Expectation: ", ""))
        score = float(label) + exp_val
        seq = seq_part.replace("Seq: ", "")

        return pd.Series([score, label, exp_val, seq])

    splitted = chunk["full"].apply(split_row)
    splitted.columns = ["Score", "Label", "Expectation", "Seq"]
    return splitted


def main():
    with open(OUTPUT_CSV, "w") as f:
        f.write("Score,Label,Expectation,Seq\n")

    total = 0
    for chunk in pd.read_csv(
        INPUT_TXT, header=None, names=["full"], chunksize=CHUNK_SIZE,
    ):
        processed = process_chunk(chunk)
        processed.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
        total += len(processed)

    print(f"CSV written to {OUTPUT_CSV}  ({total} rows)")


if __name__ == "__main__":
    main()
