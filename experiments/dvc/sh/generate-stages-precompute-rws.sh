#!/usr/bin/env bash

for dataset in 'citeseer' 'cora' 'pubmed'; do
    STAGE_FILE="experiments/dvc/stages/precompute-rws/${dataset}.dvc"

    if [ -f "${STAGE_FILE}" ]; then
        echo "Stage for ${dataset} already exists!"
        continue
    fi

    # Dependencies
    PY_SCRIPT="experiments/scripts/precompute-rws.py"
    CONFIG="experiments/configs/precompute-rws/${dataset}.yml"
    IN_DATASET_FILE="data/datasets/${dataset}.pkl"

    # Outputs
    OUT_RWS_DIR="data/rws/${dataset}/"

    # DVC command
    echo "Generating stage ${STAGE_FILE}"
    dvc run --no-exec -f "${STAGE_FILE}" \
        -d "${PY_SCRIPT}" \
        -d "${CONFIG}" \
        -d "${IN_DATASET_FILE}" \
        -o "${OUT_RWS_DIR}" \
        PYTHONPATH=. python3 "${PY_SCRIPT}" --config "${CONFIG}" --input "${IN_DATASET_FILE}" --output "${OUT_RWS_DIR}"
done
