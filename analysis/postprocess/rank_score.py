"""Filter sequences whose combined score (Label + Expectation) is exactly 4.0
and sort by Expectation descending."""

import re


def filter_max_score(input_file: str, output_file: str) -> int:
    """Find all sequences with Label + Expectation == 4.0, sort by Expectation.

    Format:  Label: {label_int} | Expectation: {exp_val} | Seq: {seq}
    """
    matched: list[tuple[float, str]] = []

    with open(input_file, "r") as f:
        for line in f:
            label_m = re.search(r"Label:\s*(\d+)", line)
            exp_m = re.search(r"Expectation:\s*([\d.\-]+)", line)
            if label_m and exp_m:
                label = int(label_m.group(1))
                exp_val = float(exp_m.group(1))
                if label == 4:  # highest class, score >= 4.0
                    matched.append((exp_val, line))

    matched.sort(key=lambda x: x[0], reverse=True)

    with open(output_file, "w") as out:
        for _, line in matched:
            out.write(line)

    print(f"Found {len(matched)} sequences with Label=4, sorted by Expectation.")
    return len(matched)


if __name__ == "__main__":
    filter_max_score(
        "./outputs/undealtotal_top_heap.txt",
        "./outputs/all_one.txt",
    )
