#!/usr/bin/env bash
# One-time environment setup on Tillicum / Hyak. Uses uv for a reproducible Python 3.12 venv
# matching the competition runtime.
set -euo pipefail

cd "$(dirname "$0")/../.."

# Ensure the official runtime submodule (score.py + build/test) is present.
git submodule update --init runtime || echo "note: run 'git submodule update --init runtime' manually"

module load cuda 2>/dev/null || echo "note: 'module load cuda' unavailable; ensure CUDA is present"

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
# CTC + KenLM decoding experiments:  uv pip install -e ".[lm]"
# Qwen3-ASR (runtime-compatible):    uv pip install -e ".[qwen]"
# Omni ASR (LOCAL RESEARCH ONLY):    uv pip install -e ".[omni-local]"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "env ready. activate with: source .venv/bin/activate"
