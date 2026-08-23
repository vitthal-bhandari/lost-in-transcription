#!/usr/bin/env bash
# One-time environment setup on Tillicum / Hyak. Uses uv for a reproducible venv.
set -euo pipefail

module load cuda 2>/dev/null || echo "note: 'module load cuda' unavailable; ensure CUDA is present"

cd "$(dirname "$0")/../.."
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
# KenLM (optional; needs a compiler toolchain). Uncomment when building n-gram LMs:
# uv pip install -e ".[lm]"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "env ready. activate with: source .venv/bin/activate"
