#!/usr/bin/env bash
# Start the isolated WSL2 Qwen3.5-4B QLoRA HTTP inference worker on localhost:8001.
set -euo pipefail
stage="${EMOTIONLAB_S4_WSL_DIR:-$HOME/.cache/emotionlab-s4-perf}"
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export TRITON_CACHE_DIR="$stage/triton-cache" TRITON_LIBCUDA_PATH=/usr/lib/wsl/lib
export HF_HOME="$stage/hf-home"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
unset PYTORCH_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$stage/venv/bin/python" -X faulthandler -u \
    -m nlp_text_classification.inference.qwen_worker "$@"
