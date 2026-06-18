#!/bin/bash
# Evaluation: Object HalBench
# See scripts/eval/README.md for dataset setup and metrics.
set -e

MODEL_PATH=${MODEL_PATH:-/path/to/tars-llava-checkpoint}
DATA_ROOT=${DATA_ROOT:-/path/to/objhal_data}
OUTPUT_DIR=${OUTPUT_DIR:-./eval_output/objhal}

mkdir -p "$OUTPUT_DIR"

# TODO(camera-ready): add inference + scoring for Object HalBench.
echo "[Object HalBench] MODEL_PATH=$MODEL_PATH DATA_ROOT=$DATA_ROOT OUTPUT_DIR=$OUTPUT_DIR"
echo "Not yet implemented — fill in the official Object HalBench protocol here."
