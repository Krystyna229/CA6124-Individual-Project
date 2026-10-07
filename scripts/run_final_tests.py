#!/usr/bin/env python3
"""Run the six final tests with the best validation parameters."""

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ml-1m", "amazon-music")
ALGORITHMS = ("ease", "multi-vae", "cdae")
STEM = "BPR_{algo}_{dataset}_10filter_tsbr_day_warm_full_opt10"


def best_parameters(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    row.pop("ndcg", None)
    return row


def main() -> None:
    for dataset in DATASETS:
        for algorithm in ALGORITHMS:
            stem = STEM.format(algo=algorithm, dataset=dataset)
            params_path = ROOT / "tune_res" / f"best_params_{stem}.csv"
            if not params_path.exists():
                raise FileNotFoundError(f"Missing tuning result: {params_path}")

            command = [
                sys.executable, "test.py",
                "--algo_name", algorithm,
                "--dataset", dataset,
                "--prepro", "10filter",
                "--test_method", "tsbr",
                "--split_boundary", "day",
                "--warm_start",
                "--ranking_mode", "full",
                "--topk", "50",
                "--save_user_metrics",
                "--loader_workers", "0",
            ]
            for name, value in best_parameters(params_path).items():
                command.extend((f"--{name}", value))

            log_path = ROOT / "experiment_logs" / f"final_{dataset}_{algorithm}.log"
            log_path.parent.mkdir(exist_ok=True)
            print(f"Running {dataset} / {algorithm}; log: {log_path}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=ROOT, stdout=log,
                               stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
