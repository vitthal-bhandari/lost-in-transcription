#!/usr/bin/env bash
# Set up a SEPARATE venv for Meta Omnilingual ASR (fairseq2 stack), isolated from the main .venv so
# it can't clobber the runtime-aligned torch/transformers pins. Also DUMPS the fully-resolved
# dependency tree to runtime_pr/omni_resolved_deps.txt — the exact list we need for the PR that
# adds omnilingual-asr to the competition runtime.
#
# Run from repo root:  bash scripts/slurm/setup_omni_env.sh
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "$(dirname "$0")/../.."
REPO_DIR="$(pwd)"

CACHE_ROOT="${LIT_CACHE_ROOT:-/gpfs/scrubbed/$USER/.cache}"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_PYTHON_INSTALL_DIR="$CACHE_ROOT/uv-python"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
# fairseq2 caches model assets (the 7B is ~30GB) under $XDG_CACHE_HOME/fairseq2 — keep off home.
export XDG_CACHE_HOME="$CACHE_ROOT"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"
echo ">>> caches -> $CACHE_ROOT (incl. XDG_CACHE_HOME for fairseq2 assets)"

case "$REPO_DIR/" in
  "$HOME"/*)
    echo "!! WARNING: repo is under \$HOME; the omni venv (fairseq2+torch, several GB) will count"
    echo "!! against your 10GB home quota. Clone under /gpfs/projects/stf/\$USER/ instead."
    read -r -p "Continue anyway? [y/N] " ok; [ "${ok:-N}" = "y" ] || exit 1 ;;
esac

VENV_DIR="${OMNI_VENV_DIR:-$REPO_DIR/.venv-omni}"
# omnilingual-asr depends on kenlm==0.3.0, a C++ source build that needs Python.h. The bare system
# /usr/bin/python3.12 has no dev headers, so force uv's MANAGED CPython (ships headers) — no root
# needed. Headers land under $UV_PYTHON_INSTALL_DIR (on scratch).
export UV_PYTHON_PREFERENCE=only-managed
uv venv --python 3.12 --python-preference only-managed "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Install omnilingual-asr FIRST so it pins its own torch/fairseq2, then add only the light deps our
# scripts need (scoring + audio + config) — no transformers, to avoid disturbing the fairseq2 stack.
uv pip install omnilingual-asr
# torch/torchaudio ABI must match. omni/fairseq2 pin torch==2.8.0, but uv pulls the LATEST
# torchaudio (2.11.0, a cu13 build that wants libcudart.so.13 — absent; only cu12 libs installed).
# Force the matching cu12 torchaudio 2.8.0, which uses libcudart.so.12 that torch 2.8.0 provides.
uv pip install "torchaudio==2.8.0"
uv pip install jiwer pandas pyarrow typer soundfile librosa pyyaml numpy

# --- capture the resolved dependency tree for the runtime PR ---
mkdir -p "$REPO_DIR/runtime_pr"
{
  echo "# Resolved by setup_omni_env.sh on $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# python: $(python -V)"
  echo "# --- uv pip freeze ---"
} > "$REPO_DIR/runtime_pr/omni_resolved_deps.txt"
uv pip freeze >> "$REPO_DIR/runtime_pr/omni_resolved_deps.txt"
echo ">>> wrote runtime_pr/omni_resolved_deps.txt"

# fairseq2n dlopens the system libsndfile.so.1 (absent on Tillicum). Expose the soundfile-bundled
# copy under a canonical name so both the import check here and the slurm runner find it.
SND=$(find "$VENV_DIR/lib" -name 'libsndfile*.so*' 2>/dev/null | head -1)
if [ -n "$SND" ]; then
  mkdir -p "$VENV_DIR/_snd"; ln -sf "$SND" "$VENV_DIR/_snd/libsndfile.so.1"
  export LD_LIBRARY_PATH="$VENV_DIR/_snd:${LD_LIBRARY_PATH:-}"
  echo ">>> libsndfile exposed for fairseq2n: $SND"
else
  echo ">>> WARN: no bundled libsndfile found under soundfile; fairseq2n import may fail."
fi

python -c "import omnilingual_asr, torch, torchaudio; print('omni OK | torch', torch.__version__, '| torchaudio', torchaudio.__version__, '| cuda', torch.cuda.is_available())" \
  || echo ">>> WARN: import check failed on this node; retry inside a GPU job before concluding."
echo ">>> omni env ready: $VENV_DIR"
