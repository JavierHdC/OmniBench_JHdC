#!/usr/bin/env python

import argparse
import os
import sys
import numpy as np

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Random baseline clustering module for the 'clustering' stage.\n"
            "Assigns random cluster labels with a fixed random seed."
        )
    )

    parser.add_argument("--data.matrix", dest="data_matrix", required=True)
    parser.add_argument("--data.true_labels", dest="data_true_labels", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    return parser.parse_args()


def load_labels(path: str) -> np.ndarray:
    arr = np.loadtxt(path)
    if arr.ndim > 1:
        arr = arr[:, 0]
    return arr.astype(int)


def count_rows(path: str) -> int:
    with open(path, "r") as fh:
        return sum(1 for _ in fh)


def main() -> int:
    args = parse_args()

    try:
        os.makedirs(args.output_dir, exist_ok=True)

        y_true = load_labels(args.data_true_labels)
        n_samples = y_true.shape[0]
        n_clusters = np.unique(y_true).size

        n_rows_matrix = count_rows(args.data_matrix)
        if n_rows_matrix != n_samples:
            raise RuntimeError(
                f"Row mismatch: matrix={n_rows_matrix}, labels={n_samples}"
            )

        rng = np.random.default_rng(seed=args.seed)
        y_pred = rng.integers(low=0, high=n_clusters, size=n_samples)

        # Renamed output here
        out_path = os.path.join(args.output_dir, f"{args.name}_JHDC_clusters.txt")
        np.savetxt(out_path, y_pred, fmt="%d")

        meta_path = os.path.join(args.output_dir, f"{args.name}_JHDC_meta.txt")
        with open(meta_path, "w") as fh:
            fh.write(f"seed: {args.seed}\n")
            fh.write(f"n_samples: {n_samples}\n")
            fh.write(f"n_clusters: {n_clusters}\n")

        print(f"Wrote random clustering to {out_path}")
        return 0

    except Exception as e:
        print(f"[random_baseline] ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
