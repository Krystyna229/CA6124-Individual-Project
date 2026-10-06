#!/usr/bin/env bash
set -euo pipefail

# Uniform protocol for all 3 algorithms x 2 datasets.
TRIALS="${TRIALS:-20}"
mkdir -p experiment_logs

for dataset in ml-1m amazon-music; do
  for algorithm in ease multi-vae cdae; do
    stem="BPR_${algorithm}_${dataset}_10filter_tsbr_day_warm_full_opt10"
    trials_file="tune_res/trials_${stem}.csv"
    if [[ -f "$trials_file" ]] && [[ $(wc -l < "$trials_file") -ge $((TRIALS + 1)) ]]; then
      echo "Skipping completed run: ${dataset}/${algorithm}"
      continue
    fi

    echo "Starting ${dataset}/${algorithm} (${TRIALS} trials)"
    python tune.py \
      --algo_name "$algorithm" \
      --dataset "$dataset" \
      --prepro 10filter \
      --test_method tsbr \
      --val_method tsbr \
      --split_boundary day \
      --warm_start \
      --ranking_mode full \
      --topk 50 \
      --optimization_metric ndcg \
      --optimization_k 10 \
      --loader_workers 0 \
      --study_storage sqlite:///tune_res/assignment_studies.db \
      --study_name "${dataset}_${algorithm}_expanded" \
      --hyperopt_trail "$TRIALS" \
      >"experiment_logs/tune_${dataset}_${algorithm}.log" 2>&1
    echo "Finished ${dataset}/${algorithm}"
  done
done
