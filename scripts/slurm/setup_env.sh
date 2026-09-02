#!/usr/bin/env bash
# One-time environment setup on Tillicum. Creates a reproducible Python 3.12 venv (matching the
# competition runtime) and redirects EVERY package/model cache to scratch so the 10GB home quota
# is never touched.
#
# Run from the repo root:  bash scripts/slurm/setup_env.sh
set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "$(dirname "$0")/../.."
REPO_DIR="$(pwd)"

# --- caches on scratch (re-downloadable; scratch may be auto-purged, that's fine) ---------------
# Override LIT_CACHE_ROOT if your scratch path differs (e.g. /gscratch/... on some clusters).
CACHE_ROOT="${LIT_CACHE_ROOT:-/gpfs/scrubbed/$USER/.cache}"
export UV_CACHE_DIR="$CACHE_ROOT/uv"                     # uv wheel/build cache
export UV_PYTHON_INSTALL_DIR="$CACHE_ROOT/uv-python"     # the managed 3.12 interpreter itself
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/huggingface"                 # SAME path the train/eval slurm use
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$CACHE_ROOT/torch"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PIP_CACHE_DIR" "$HF_HUB_CACHE" "$TORCH_HOME"

echo ">>> caches -> $CACHE_ROOT"

# --- guard: the venv (several GB with torch) must NOT live under the 10GB home quota ------------
# The venv lives IN the repo, so the repo must be on projects/scratch, not home.
case "$REPO_DIR/" in
  "$HOME"/*)
    echo "!! WARNING: repo is under \$HOME ($REPO_DIR)."
    echo "!! The .venv (torch etc. ~5-8GB) will count against your 10GB home quota."
    echo "!! Recommended: clone under /gpfs/projects/stf/\$USER/ and re-run this script there."
    echo "!! (Set VENV_DIR=/some/scratch/or/projects/path to put the venv elsewhere, if you must.)"
    read -r -p "Continue anyway? [y/N] " ok; [ "${ok:-N}" = "y" ] || exit 1
    ;;
esac

# --- submodule + venv --------------------------------------------------------------------------
git submodule update --init runtime || echo "note: run 'git submodule update --init runtime' manually"

VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
uv venv --python 3.12 "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv pip install -e ".[dev]"
# CTC + KenLM decoding experiments:  uv pip install -e ".[lm]"
# Qwen3-ASR (runtime-compatible):    uv pip install -e ".[qwen]"
# Omni ASR (LOCAL RESEARCH ONLY):    uv pip install -e ".[omni-local]"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo ">>> env ready. venv: $VENV_DIR"
echo ">>> activate with: source $VENV_DIR/bin/activate"
echo ">>> NOTE: cuda=False is EXPECTED on a login node; slurm jobs get the GPU."
