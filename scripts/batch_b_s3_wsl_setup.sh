#!/usr/bin/env bash
# Run from the EmotionLab checkout in the existing Ubuntu WSL2 distribution.
set -euo pipefail
stage="${EMOTIONLAB_S3_WSL_DIR:-$HOME/.cache/emotionlab-s3-perf}"
uv="${UV_BIN:-$HOME/.local/bin/uv}"
mkdir -p "$stage"
export UV_CACHE_DIR="$stage/uv-cache"
python="$stage/venv/bin/python"

if [[ ! -x "$python" ]]; then
    "$uv" venv --python 3.12 "$stage/venv"
fi
"$uv" pip install --python "$python" --index-url https://download.pytorch.org/whl/cu128 \
    'torch==2.9.1+cu128'
"$python" -c 'import torch; assert torch.cuda.is_available() and torch.compiled_with_cxx11_abi()'

wheel='causal_conv1d-1.7.0+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
wheel_digest='506b7a7c7d89790340b99bac53be5a292237d71d4ac9f1ef1ac5ad98cdcc30c1'
if [[ ! -f "$stage/$wheel" ]]; then
    curl --fail --location --retry 3 --continue-at - \
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/$wheel" \
        --output "$stage/$wheel.partial"
    mv "$stage/$wheel.partial" "$stage/$wheel"
fi
printf '%s  %s\n' "$wheel_digest" "$stage/$wheel" | sha256sum --check -
awk '$1 != "causal-conv1d"' configs/batch_b/s3-wsl-runtime.lock.txt > "$stage/constraints.txt"
"$uv" pip install --python "$python" --constraint "$stage/constraints.txt" \
    'transformers==5.10.2' 'peft==0.19.0' 'accelerate==1.16.0' 'bitsandbytes==0.49.2' \
    'tokenizers==0.22.2' 'safetensors==0.8.0' numpy scikit-learn PyYAML psutil 'optuna==5.1.0' \
    'flash-linear-attention[cuda]==0.5.1' "$stage/$wheel"
"$uv" pip check --python "$python"
"$uv" pip freeze --python "$python" > artifacts/batch-b-s3/performance/wsl-runtime.lock.txt
"$python" - <<'PY'
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
print("DeltaNet kernels:", qwen.chunk_gated_delta_rule, qwen.fused_recurrent_gated_delta_rule)
print("Convolution kernels:", qwen.causal_conv1d_fn, qwen.causal_conv1d_update)
assert qwen.is_fast_path_available
PY
