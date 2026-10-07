#!/usr/bin/env python3
"""Paired bootstrap confidence intervals for final per-user NDCG@10."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ml-1m", "amazon-music")
ALGORITHMS = ("ease", "multi-vae", "cdae")
DISPLAY = {"ease": "EASE", "multi-vae": "Multi-VAE", "cdae": "CDAE"}
SEED = 2022
N_BOOTSTRAP = 10_000


def load_scores(dataset: str, algorithm: str) -> pd.DataFrame:
    path = (ROOT / "res" / dataset / "10filter" / "tsbr" /
            f"BPR_{algorithm}_day_warm_full_with_0uniform_user_ndcg10.csv")
    frame = pd.read_csv(path)
    if len(frame.columns) != 2:
        raise ValueError(f"Expected user and ndcg@10 columns in {path}")
    return frame.rename(columns={frame.columns[1]: algorithm})


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for dataset in DATASETS:
        frames = [load_scores(dataset, algorithm) for algorithm in ALGORITHMS]
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on=merged.columns[0], validate="one_to_one")
        if any(len(frame) != len(merged) for frame in frames):
            raise ValueError(f"Per-user cohorts do not match for {dataset}")

        for left, right in combinations(ALGORITHMS, 2):
            paired_diff = (merged[left] - merged[right]).to_numpy(dtype=float)
            indices = rng.integers(0, len(paired_diff),
                                  size=(N_BOOTSTRAP, len(paired_diff)))
            bootstrap_means = paired_diff[indices].mean(axis=1)
            ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
            # Two-sided bootstrap sign probability; the +1 avoids a zero estimate.
            p_value = 2 * min(
                (np.count_nonzero(bootstrap_means <= 0) + 1) / (N_BOOTSTRAP + 1),
                (np.count_nonzero(bootstrap_means >= 0) + 1) / (N_BOOTSTRAP + 1),
            )
            rows.append({
                "dataset": dataset,
                "comparison": f"{DISPLAY[left]} - {DISPLAY[right]}",
                "n_users": len(paired_diff),
                "mean_difference": paired_diff.mean(),
                "ci_2.5%": ci_low,
                "ci_97.5%": ci_high,
                "bootstrap_p": min(p_value, 1.0),
                "seed": SEED,
                "resamples": N_BOOTSTRAP,
            })

    output = ROOT / "results" / "paired_bootstrap_ndcg10.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
