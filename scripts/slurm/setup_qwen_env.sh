#!/usr/bin/env bash
# Isolated venv for Qwen3-ASR (qwen-asr package), separate from the main .venv so its torch/
# transformers pins can't disturb the validated Whisper stack. qwen-asr is SUBMITTABLE (it's in
# the competition runtime), so this is a real candidate base, not just research.
#
# Run from repo root:  bash scripts/slurm/setup_qwen_env.sh
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "$(dirname "$0")/../.."
REPO_DIR="$(pwd)"

CACHE_ROOT="${LIT_CACHE_ROOT:-/gpfs/scrubbed/$USER/.cache}"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_PYTHON_INSTALL_DIR="$CACHE_ROOT/uv-python"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/huggingface"          # Qwen weights download here (~a few GB)
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PIP_CACHE_DIR" "$HF_HUB_CACHE"
echo ">>> caches -> $CACHE_ROOT"

case "$REPO_DIR/" in
  "$HOME"/*)
    echo "!! WARNING: repo under \$HOME; the qwen venv (torch, several GB) hits your 10GB home quota."
    echo "!! Clone under /gpfs/projects/stf/\$USER/ instead."
    read -r -p "Continue anyway? [y/N] " ok; [ "${ok:-N}" = "y" ] || exit 1 ;;
esac

VENV_DIR="${QWEN_VENV_DIR:-$REPO_DIR/.venv-qwen}"
# Managed CPython (ships headers) in case any dep builds from source; no root needed.
export UV_PYTHON_PREFERENCE=only-managed
uv venv --python 3.12 --python-preference only-managed "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# qwen-asr pulls its own torch/transformers; then add the light deps our scripts need.
uv pip install qwen-asr
uv pip install jiwer pandas pyarrow soundfile librosa numpy pyyaml typer

mkdir -p "$REPO_DIR/runtime_pr"
{ echo "# qwen-asr resolved on $(hostname) $(date -u +%Y-%m-%dT%H:%M:%SZ) | $(python -V)";
  uv pip freeze; } > "$REPO_DIR/runtime_pr/qwen_resolved_deps.txt"
echo ">>> wrote runtime_pr/qwen_resolved_deps.txt"

# NOTE: qwen-asr does NOT depend on torchaudio (uses sox/soxr/scipy); don't import it here.
python -c "import qwen_asr, torch; print('qwen OK | torch', torch.__version__, '| cuda', torch.cuda.is_available())" \
  || echo ">>> WARN: import check failed on this node; retry inside a GPU job before concluding."
echo ">>> qwen env ready: $VENV_DIR"
