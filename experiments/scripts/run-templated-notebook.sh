#!/usr/bin/env bash

NOTEBOOK_FILE="${1}"
CONFIG_FILE="${2}"
OUTPUT_PATH="${3}"

OUTPUT_FILE="$(basename ${OUTPUT_PATH})"
OUTPUT_DIR="$(dirname ${OUTPUT_PATH})"

papermill --prepare-only -f "${CONFIG_FILE}" "${NOTEBOOK_FILE}" Notebook.ipynb

jupyter nbconvert --to html \
        --execute Notebook.ipynb \
        --template experiments/notebooks/hidecode.tpl \
        --output "${OUTPUT_FILE}" \
        --output-dir "${OUTPUT_DIR}" \
        --ExecutePreprocessor.timeout=-1

rm -f Notebook.ipynb
