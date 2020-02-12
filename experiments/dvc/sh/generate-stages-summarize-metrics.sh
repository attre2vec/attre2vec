#!/usr/bin/env bash

STAGE_FILE="experiments/dvc/stages/summarize-metrics.dvc"

if [ -f "${STAGE_FILE}" ]; then
    echo "Stage ${STAGE_FILE} already exists!"
    continue
fi

# Dependencies
SH_SCRIPT='experiments/scripts/run-templated-notebook.sh'
NOTEBOOK_FILE="experiments/notebooks/summarize-metrics.ipynb"
HIDECODE_TPL="experiments/notebooks/hidecode.tpl"
CONFIG="experiments/configs/summarize-metrics.yml"
METRICS_DIR="data/metrics/"

# Outputs
HTML_RESULTS_FILE="data/plots/metrics/summary.html"

# DVC command
echo "Generating stage ${STAGE_FILE}"
dvc run --no-exec -f "${STAGE_FILE}" \
    -d "${SH_SCRIPT}" \
    -d "${NOTEBOOK_FILE}" \
    -d "${HIDECODE_TPL}" \
    -d "${CONFIG}" \
    -d "${METRICS_DIR}" \
    -o "${HTML_RESULTS_FILE}" \
    bash -c "${SH_SCRIPT} ${NOTEBOOK_FILE} ${CONFIG} ${HTML_RESULTS_FILE}"
