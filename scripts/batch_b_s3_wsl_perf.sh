#!/usr/bin/env bash
# Run from the EmotionLab checkout after creating the matching Windows initial adapter.
set -euo pipefail
stage="${EMOTIONLAB_S3_WSL_DIR:-$HOME/.cache/emotionlab-s3-perf}"
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export TRITON_CACHE_DIR="$stage/triton-cache" TRITON_LIBCUDA_PATH=/usr/lib/wsl/lib
export HF_HOME="$stage/hf-home"
exec "$stage/venv/bin/python" -X faulthandler -u \
    -m nlp_text_classification.experiments.batch_b_s3_perf wsl "$@"
