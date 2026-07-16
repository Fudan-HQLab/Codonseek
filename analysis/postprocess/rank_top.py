"""Extract top-100 sequences from the combined top-heap file, ranked by
combined score (Label + Expectation) descending, then Expectation descending."""

import re
from heapq import nlargest


def extract_top100(input_file: str, output_file: str) -> int:
    """Parse sequences from *input_file*, select top 100 by score, write TSV."""

    # Format:  Label: {label_int} | Expectation: {exp_val} | Seq: {seq}
    pattern = re.compile(
        r"Label:\s*(\d+)\s*\|\s*Expectation:\s*([\d.\-]+)\s*\|\s*Seq:\s*(.*)$"
    )

    records: list[dict] = []
    line_count = 0
    valid_count = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            line = line.strip()
            if not line or "====" in line or "From:" in line:
                continue

            m = pattern.match(line)
            if m:
                try:
                    label = int(m.group(1))
                    exp_val = float(m.group(2))
                    seq = m.group(3).strip()
                    score = float(label) + exp_val
                    records.append({
                        "label": label,
                        "expectation": exp_val,
                        "score": score,
                        "seq": seq,
                        "original": line,
                    })
                    valid_count += 1
                except ValueError:
                    pass

    print(f"Parsed {valid_count} valid records from {line_count} lines.")

    if not records:
        print("No valid records found.")
        return 0

    top100 = nlargest(100, records, key=lambda r: (r["score"], r["expectation"]))

    with open(output_file, "w", encoding="utf-8") as out:
        out.write("Rank\tScore\tLabel\tExpectation\tSequence\n")
        out.write("=" * 100 + "\n")
        for rank, rec in enumerate(top100, 1):
            out.write(
                f"{rank}\t{rec['score']:.4f}\t{rec['label']}\t"
                f"{rec['expectation']:.4f}\t{rec['seq']}\n"
            )

    print(f"Top 100 written to {output_file}")
    if top100:
        print(f"Score range: {top100[0]['score']:.4f} -- {top100[-1]['score']:.4f}")
    return len(top100)


if __name__ == "__main__":
    extract_top100(
        "./outputs/undealtotal_top_heap.txt",
        "./outputs/top100_results.txt",
    )
