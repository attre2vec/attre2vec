#!/usr/bin/env bash

for dataset in 'citeseer' 'cora' 'pubmed'; do
    STAGE_FILE="experiments/dvc/stages/eval-line2vec/${dataset}.dvc"

    if [ -f "${STAGE_FILE}" ]; then
        echo "Stage ${STAGE_FILE} already exists!"
        continue
    fi

    # Dependencies
    PY_SCRIPT="experiments/scripts/eval-line2vec.py"
    CONFIG="experiments/configs/eval-line2vec/${dataset}.yml"
    IN_DATASET_FILE="data/datasets/${dataset}.pkl"
    IN_EMB_FILE="data/line2vec/embed/${dataset}.pkl"

    # Outputs
    OUT_VECTORS_DIR="data/vectors/${dataset}/BL_line2vec/"
    OUT_METRICS_FILE="data/metrics/bl/${dataset}/line2vec.pkl"

    # DVC command
    echo "Generating stage ${STAGE_FILE}"
    dvc run --no-exec -f "${STAGE_FILE}" \
        -d "${PY_SCRIPT}" \
        -d "${CONFIG}" \
        -d "${IN_DATASET_FILE}" \
        -d "${IN_EMB_FILE}" \
        -o "${OUT_VECTORS_DIR}" \
        -o "${OUT_METRICS_FILE}" \
        PYTHONPATH=. python3 "${PY_SCRIPT}" --config "${CONFIG}"
done