#!/bin/bash
# Evaluation: MME
# See scripts/eval/README.md for dataset setup and metrics.
set -e

MODEL_PATH=${MODEL_PATH:-/path/to/tars-llava-checkpoint}
DATA_ROOT=${DATA_ROOT:-/path/to/mme_data}
OUTPUT_DIR=${OUTPUT_DIR:-./eval_output/mme}

mkdir -p "$OUTPUT_DIR"

# TODO(camera-ready): add inference + scoring for MME.
echo "[MME] MODEL_PATH=$MODEL_PATH DATA_ROOT=$DATA_ROOT OUTPUT_DIR=$OUTPUT_DIR"
echo "Not yet implemented — fill in the official MME protocol here."
