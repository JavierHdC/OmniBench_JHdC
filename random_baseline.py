#!/usr/bin/env python

import argparse
import os
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Random baseline clustering module for the 'clustering' stage.\n\n"
            "Assigns random cluster labels with the same number of clusters "
            "as in --data.true_labels."
        )
    )

    # Stage-specific arguments
    parser.add_argument(
        "--data.matrix",
        dest="data_matrix",
        required=True,
        help="Path to the feature matrix file for clustering.",
    )
    parser.add_argument(
        "--data.true_labels",
        dest="data_true_labels",
        required=True,
        help="Path to the file with true labels (one label per row).",
    )

    # Generic stage arguments
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Name of the module (used in output filenames).",
    )

    # Seed control
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible baseline clustering (default: 42).",
    )

    return parser.parse_args()


def load_labels(path: str) -> np.ndarray:
    """
    Load labels from a text file. Assumes one label per line.
    Extra columns are ignored.
    """
    try:
        arr = np.loadtxt(path)
    except Exception as e:
        raise RuntimeError(f"Could not load true labels from {path}: {e}")

    # If 2D, take first column
    if arr.ndim > 1:
        arr = arr[:, 0]

    return arr.astype(int)


def count_rows(path: str) -> int:
    """
    Count number of observations from the data matrix file.
    We don't use the actual feature values here, only the number of rows.
    """
    try:
        with open(path, "r") as fh:
            return sum(1 for _ in fh)
    except Exception as e:
        raise RuntimeError(f"Could not read data matrix from {path}: {e}")


def main() -> int:
    args = parse_args()

    try:
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)

        # Load true labels and infer number of clusters
        y_true = load_labels(args.data_true_labels)
        n_samples = y_true.shape[0]
        unique_labels = np.unique(y_true)
        n_clusters = unique_labels.size

        # Sanity check: data.matrix should have same number of rows
        n_rows_matrix = count_rows(args.data_matrix)
        if n_rows_matrix != n_samples:
            raise RuntimeError(
                f"Row count mismatch between data.matrix ({n_rows_matrix}) "
                f"and data.true_labels ({n_samples})."
            )

        # Initialise RNG with fixed seed
        rng = np.random.default_rng(seed=args.seed)

        # Draw random cluster labels 0..(k-1)
        y_pred = rng.integers(low=0, high=n_clusters, size=n_samples)

        # Save predictions
        out_path = os.path.join(args.output_dir, f"{args.name}_clusters.txt")
        np.savetxt(out_path, y_pred, fmt="%d")

        # Optional: store metadata (seed, k, etc.) for debugging
        meta_path = os.path.join(args.output_dir, f"{args.name}_meta.txt")
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
