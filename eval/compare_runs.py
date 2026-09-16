"""Compare evaluation runs grouped by label (mean ± std across repeated runs).

LLM-judge scores vary by several points between runs, so compare configurations
over repeated runs rather than single ones.

Usage:
    python eval/compare_runs.py                 # all labelled runs in eval/results/
    python eval/compare_runs.py ctx500 ctxfull  # only these labels, in this order
"""

import glob
import json
import os
import statistics
import sys
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

COLUMNS = [
    ("metrics", "faithfulness"),
    ("metrics", "answer_relevancy"),
    ("metrics", "context_precision"),
    ("metrics", "context_recall"),
    ("retrieval", "hit_rate@5"),
    ("retrieval", "mrr"),
]


def load_runs() -> dict[str, list[dict]]:
    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*", "summary.json"))):
        folder = os.path.basename(os.path.dirname(path))
        # Folder format: YYYYmmdd-HHMMSS[-label]
        label = folder.split("-", 2)[2] if folder.count("-") >= 2 else "unlabelled"
        with open(path, encoding="utf-8") as f:
            runs[label].append(json.load(f))
    return runs


def fmt(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def main():
    runs = load_runs()
    labels = sys.argv[1:] or sorted(runs)

    header = ["label", "runs"] + [name for _, name in COLUMNS]
    table = [header]
    for label in labels:
        summaries = runs.get(label, [])
        if not summaries:
            print(f"No runs found for label '{label}'")
            continue
        row = [label, str(len(summaries))]
        for group, name in COLUMNS:
            row.append(fmt([s[group][name] for s in summaries]))
        table.append(row)

    widths = [max(len(r[i]) for r in table) for i in range(len(header))]
    for i, row in enumerate(table):
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))
        if i == 0:
            print("  ".join("-" * w for w in widths))


if __name__ == "__main__":
    main()
