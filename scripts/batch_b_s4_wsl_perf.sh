#!/usr/bin/env bash
# Run Batch B S4 performance investigation in the isolated WSL2 runtime.
set -euo pipefail
s3="${EMOTIONLAB_S3_WSL_DIR:-$HOME/.cache/emotionlab-s3-perf}"
stage="${EMOTIONLAB_S4_WSL_DIR:-$HOME/.cache/emotionlab-s4-perf}"
mkdir -p "$stage/triton-cache" "$stage/hf-home"
if [[ -d "$s3/triton-cache" && -z "$(ls -A "$stage/triton-cache")" ]]; then
    cp -a "$s3/triton-cache/." "$stage/triton-cache/"
fi
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export TRITON_CACHE_DIR="$stage/triton-cache" TRITON_LIBCUDA_PATH=/usr/lib/wsl/lib
export HF_HOME="$stage/hf-home"
exec "$stage/venv/bin/python" -X faulthandler -u \
    -m nlp_text_classification.experiments.batch_b_s4_perf "$@"
