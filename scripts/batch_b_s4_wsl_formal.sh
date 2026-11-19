#!/usr/bin/env bash
# Use the isolated S4 runtime for bounded acceptance and formal training.
set -euo pipefail
stage="${EMOTIONLAB_S4_WSL_DIR:-$HOME/.cache/emotionlab-s4-perf}"
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export TRITON_CACHE_DIR="$stage/triton-cache" TRITON_LIBCUDA_PATH=/usr/lib/wsl/lib
export HF_HOME="$stage/hf-home"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
# Torch 2.9.1 enabled expandable CUDA segments through this variable in WSL2.
unset PYTORCH_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$stage/venv/bin/python" -X faulthandler -u \
    -m nlp_text_classification.experiments.batch_b_s4_formal_watchdog "$@"
