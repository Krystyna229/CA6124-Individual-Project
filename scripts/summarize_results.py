#!/usr/bin/env python3
"""Create compact final-result tables and figures for the report."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
DATASETS = ("ml-1m", "amazon-music")
ALGORITHMS = ("ease", "multi-vae", "cdae")
DISPLAY = {"ease": "EASE", "multi-vae": "Multi-VAE", "cdae": "CDAE"}
METRICS = ("Recall", "NDCG")


def read_metric(dataset: str, algorithm: str, metric: str, k: int = 10) -> float:
    path = (ROOT / "res" / dataset / "10filter" / "tsbr" /
            f"BPR_{algorithm}_day_warm_full_with_0uniform_kpi_results.csv")
    frame = pd.read_csv(path, index_col="KPI@K")
    return float(frame.loc[metric, str(k)])


def final_table() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for algorithm in ALGORITHMS:
            row = {"dataset": dataset, "model": DISPLAY[algorithm]}
            row.update({f"{m.lower()}@10": read_metric(dataset, algorithm, m)
                        for m in METRICS})
            rows.append(row)
    return pd.DataFrame(rows)


def result_figure(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    x = np.arange(len(ALGORITHMS))
    width = 0.35
    for axis, dataset in zip(axes, DATASETS):
        subset = frame[frame.dataset == dataset].set_index("model")
        for offset, metric, label in ((-width / 2, "ndcg@10", "NDCG@10"),
                                      (width / 2, "recall@10", "Recall@10")):
            values = [float(subset.loc[DISPLAY[a], metric]) for a in ALGORITHMS]
            axis.bar(x + offset, values, width, label=label)
        axis.set_xticks(x, [DISPLAY[a] for a in ALGORITHMS])
        axis.set_title(dataset)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Score")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "final_results.pdf", bbox_inches="tight")
    fig.savefig(OUT / "final_results.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def tuning_figure() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 4.6), sharex=True)
    for row, dataset in enumerate(DATASETS):
        for col, algorithm in enumerate(ALGORITHMS):
            path = ROOT / "tune_res" / (
                f"trials_BPR_{algorithm}_{dataset}_10filter_tsbr_"
                "day_warm_full_opt10.csv")
            trials = pd.read_csv(path).sort_values("number")
            complete = trials[trials.state == "COMPLETE"]
            best = complete.value.cummax()
            ax = axes[row, col]
            ax.plot(complete.number + 1, best, linewidth=1.8)
            ax.scatter(complete.number + 1, complete.value, s=8, alpha=.3)
            ax.set_title(f"{dataset} · {DISPLAY[algorithm]}", fontsize=9)
            ax.grid(alpha=.2)
            if col == 0:
                ax.set_ylabel("Validation NDCG@10")
            if row == 1:
                ax.set_xlabel("Trial")
    fig.tight_layout()
    fig.savefig(OUT / "tuning_curves.pdf", bbox_inches="tight")
    fig.savefig(OUT / "tuning_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def best_parameter_table() -> None:
    rows = []
    for dataset in DATASETS:
        for algorithm in ALGORITHMS:
            path = ROOT / "tune_res" / (
                f"best_params_BPR_{algorithm}_{dataset}_10filter_tsbr_"
                "day_warm_full_opt10.csv")
            with path.open(newline="", encoding="utf-8") as handle:
                result = next(csv.DictReader(handle))
            rows.append({"dataset": dataset, "model": DISPLAY[algorithm], **result})
    pd.DataFrame(rows).to_csv(OUT / "best_parameters.csv", index=False)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    frame = final_table()
    frame.to_csv(OUT / "final_summary.csv", index=False)
    best_parameter_table()
    result_figure(frame)
    tuning_figure()


if __name__ == "__main__":
    main()
