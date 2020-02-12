#!/usr/bin/env bash

for dataset in 'citeseer' 'cora' 'pubmed'; do
    for model in 'AttrE2vec_Avg' 'AttrE2vec_Exp' 'AttrE2vec_ConcatGRU' 'AttrE2vec_GRU'; do
        STAGE_FILE="experiments/dvc/stages/train-eval-model/${dataset}/${model}.dvc"

        if [ -f "${STAGE_FILE}" ]; then
            echo "Stage ${STAGE_FILE} already exists!"
            continue
        fi

        # Dependencies
        PY_SCRIPT="experiments/scripts/train-eval-model.py"
        COMMON_CONFIG="experiments/configs/train-eval-model/${dataset}/common.yml"
        MODEL_CONFIG="experiments/configs/train-eval-model/${dataset}/${model}.yml"
        IN_DATASET_FILE="data/datasets/${dataset}.pkl"
        IN_RWS_DIR="data/rws/${dataset}/"

        # Outputs
        OUT_MODELS_DIR="data/models/${dataset}/${model}/"
        OUT_LOSSES_DIR="data/plots/losses/${dataset}/${model}/"
        OUT_LOGS_DIR="data/logs/${dataset}/${model}/"
        OUT_VECTORS_DIR="data/vectors/${dataset}/${model}/"
        OUT_METRICS_FILE="data/metrics/ae/${dataset}/${model}.pkl"

        # DVC command
        echo "Generating stage ${STAGE_FILE}"
        dvc run --no-exec -f "${STAGE_FILE}" \
            -d "${PY_SCRIPT}" \
            -d "${COMMON_CONFIG}" \
            -d "${MODEL_CONFIG}" \
            -d "${IN_DATASET_FILE}" \
            -d "${IN_RWS_DIR}" \
            -o "${OUT_MODELS_DIR}" \
            -o "${OUT_LOSSES_DIR}" \
            -o "${OUT_LOGS_DIR}" \
            -o "${OUT_METRICS_FILE}" \
            -o "${OUT_VECTORS_DIR}" \
            PYTHONPATH=. python3 "${PY_SCRIPT}" --common-config "${COMMON_CONFIG}" --model-config "${MODEL_CONFIG}"
    done
done
