#!/usr/bin/env python

import argparse
import gzip
import os
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Random baseline clustering module for the 'clustering' stage.\n"
            "Assigns random cluster labels with the same number of clusters "
            "as in --data.true_labels."
        )
    )

    parser.add_argument("--data.matrix", dest="data_matrix", required=True)
    parser.add_argument("--data.true_labels", dest="data_true_labels", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def smart_open(path: str, mode: str = "rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def load_labels(path: str) -> np.ndarray:
    try:
        with smart_open(path, "rt") as f:
            arr = np.loadtxt(f)
    except Exception as e:
        raise RuntimeError(f"Could not load true labels from {path}: {e}")

    if arr.ndim > 1:
        arr = arr[:, 0]
    return arr.astype(int)


def count_rows(path: str) -> int:
    try:
        with smart_open(path, "rt") as fh:
            return sum(1 for _ in fh)
    except Exception as e:
        raise RuntimeError(f"Could not read data matrix from {path}: {e}")


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
                f"Row count mismatch between data.matrix ({n_rows_matrix}) "
                f"and data.true_labels ({n_samples})."
            )

        rng = np.random.default_rng(seed=args.seed)
        y_pred = rng.integers(low=0, high=n_clusters, size=n_samples)

        # 1) main file that Snakemake expects
        ks_path = os.path.join(args.output_dir, f"{args.name}_ks_range.labels.gz")
        with gzip.open(ks_path, "wt") as f:
            np.savetxt(f, y_pred, fmt="%d")

        # 2) your extra JHDC debugging file (optional but keeps your naming)
        jhdc_path = os.path.join(args.output_dir, f"{args.name}_JHDC_clusters.txt")
        np.savetxt(jhdc_path, y_pred, fmt="%d")

        meta_path = os.path.join(args.output_dir, f"{args.name}_JHDC_meta.txt")
        with open(meta_path, "w") as fh:
            fh.write(f"seed: {args.seed}\n")
            fh.write(f"n_samples: {n_samples}\n")
            fh.write(f"n_clusters: {n_clusters}\n")

        print(f"Wrote random clustering to {ks_path}")
        return 0

    except Exception as e:
        print(f"[random_baseline] ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
