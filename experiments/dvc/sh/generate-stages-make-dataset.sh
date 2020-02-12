#!/usr/bin/env bash

for dataset in 'citeseer' 'cora' 'pubmed'; do
    STAGE_FILE="experiments/dvc/stages/make-dataset/${dataset}.dvc"

    if [ -f "${STAGE_FILE}" ]; then
        echo "Stage for ${dataset} already exists!"
        continue
    fi

    # Dependencies
    PY_SCRIPT="experiments/scripts/make-dataset.py"
    CONFIG="experiments/configs/make-dataset/${dataset}.yml"

    # Outputs
    OUT_DATASET_FILE="data/datasets/${dataset}.pkl"

    # DVC command
    echo "Generating stage ${STAGE_FILE}"
    dvc run --no-exec -f "${STAGE_FILE}" \
        -d "${PY_SCRIPT}" \
        -d "${CONFIG}" \
        -o "${OUT_DATASET_FILE}" \
        PYTHONPATH=. python3 "${PY_SCRIPT}" --config "${CONFIG}" --name "${dataset}" --output "${OUT_DATASET_FILE}"
done
