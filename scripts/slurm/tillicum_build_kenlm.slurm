#!/bin/bash
#SBATCH --job-name=lit-kenlm
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Build a KenLM from a manifest's training text (CPU-only work; GPU requested only because
# Tillicum requires --gres=gpu on every job).
# Usage:
#   sbatch --export=ALL,MANIFEST=data/manifests/id_jv_plus_homostoria.parquet,\
#          OUT=data/lm/id_jv_plus_homostoria,ORDER=4 scripts/slurm/tillicum_build_kenlm.slurm
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"

[ -x tools/kenlm/bin/lmplz ] || bash scripts/slurm/setup_kenlm_tools.sh
python -c "import kenlm" 2>/dev/null || uv pip install -q kenlm
python -c "import pyctcdecode" 2>/dev/null || uv pip install -q pyctcdecode

MANIFEST=${MANIFEST:?set MANIFEST=data/manifests/...parquet}
OUT=${OUT:?set OUT=data/lm/<name>}
ORDER=${ORDER:-4}
SPLITS=${SPLITS:-train,val}

python scripts/build_kenlm.py --manifest "$MANIFEST" --splits "$SPLITS" --order "$ORDER" --out "$OUT"
