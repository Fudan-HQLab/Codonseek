"""Combine all mcts_top_heap and selfplay_top_heap files into a single text file.

This script runs inside `outputs/` (cwd = project root) and reads from
`outputs/mcts_top_heap/` and `outputs/selfplay_top_heap/`.
"""

import glob
import os


TOP_HEAP_DIRS = [
    "./outputs/mcts_top_heap",
    "./outputs/selfplay_top_heap",
]


def merge_top_heap_files(output_file: str = "./outputs/combined_texts.txt") -> int:
    """Collect every `*.txt` file from the top-heap directories and
    concatenate them into *output_file*.  Returns the number of files merged."""
    all_paths: list[str] = []
    for d in TOP_HEAP_DIRS:
        if os.path.isdir(d):
            paths = glob.glob(os.path.join(d, "*.txt"))
            all_paths.extend(paths)

    all_paths.sort()

    if not all_paths:
        print("No top-heap files found.")
        return 0

    total_lines = 0
    total_chars = 0

    with open(output_file, "w", encoding="utf-8") as out:
        for file_path in all_paths:
            fname = os.path.basename(file_path)
            out.write(f"\n{'=' * 50}\nFrom: {fname}\n{'=' * 50}\n\n")

            with open(file_path, "r", encoding="utf-8") as inf:
                content = inf.read()
                out.write(content)
                total_lines += content.count("\n") + 1
                total_chars += len(content)

    print(f"Merged {len(all_paths)} files -> {output_file}  "
          f"({total_lines} lines, {total_chars} chars)")
    return len(all_paths)


if __name__ == "__main__":
    merge_top_heap_files()
