#!/usr/bin/env bash

for dataset in 'citeseer' 'cora' 'pubmed'; do
    for baseline in 'simple' 'dw' 'n2v' 'sdne' 'struc2vec' 'graphsage'; do
      STAGE_FILE="experiments/dvc/stages/eval-naive-baseline/${dataset}/${baseline}.dvc"

      if [ -f "${STAGE_FILE}" ]; then
          echo "Stage for ${STAGE_FILE} already exists!"
          continue
      fi

      # Dependencies
      PY_SCRIPT="experiments/scripts/eval-naive-baseline.py"
      COMMON_CONFIG="experiments/configs/eval-naive-baseline/${dataset}/common.yml"
      CONFIG="experiments/configs/eval-naive-baseline/${dataset}/${baseline}.yml"
      IN_DATASET_FILE="data/datasets/${dataset}.pkl"

      # Outputs
      OUT_VECTORS_DIR="data/vectors/${dataset}/BL_${baseline}/"
      OUT_METRICS_DIR="data/metrics/bl/${dataset}/${baseline}/"

      # DVC command
      echo "Generating stage ${STAGE_FILE}"
      dvc run --no-exec -f "${STAGE_FILE}" \
          -d "${PY_SCRIPT}" \
          -d "${COMMON_CONFIG}" \
          -d "${CONFIG}" \
          -d "${IN_DATASET_FILE}" \
          -o "${OUT_VECTORS_DIR}" \
          -o "${OUT_METRICS_DIR}" \
          PYTHONPATH=. python3 "${PY_SCRIPT}" --common-config ${COMMON_CONFIG} --config "${CONFIG}"
  done
done