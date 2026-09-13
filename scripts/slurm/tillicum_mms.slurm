#!/bin/bash
#SBATCH --job-name=lit-mms
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Zero-shot MMS-1B-all benchmark (greedy or +KenLM beam search) on Tillicum H200. Main .venv
# (transformers already present; pyctcdecode/kenlm come from the "lm" extra).
# Usage:
#   sbatch --export=ALL,TRACK=id_jv,RUN=mms_greedy scripts/slurm/tillicum_mms.slurm
#   sbatch --export=ALL,TRACK=id_jv,KENLM=data/lm/id_jv_plus_homostoria/lm_4gram.bin,\
#          RUN=mms_kenlm scripts/slurm/tillicum_mms.slurm
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="${LIT_CACHE_ROOT:-/gpfs/scrubbed/$USER/.cache}/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE"

python -c "import pyctcdecode" 2>/dev/null || uv pip install -q -e ".[lm]"

TRACK=${TRACK:?set TRACK}
SPLIT=${SPLIT:-dev}
MODEL=${MODEL:-facebook/mms-1b-all}
LANGCODE=${LANGCODE:-ind}
KENLM=${KENLM:-}
RUN=${RUN:-mms}
MANIFEST=${MANIFEST:-data/manifests/${TRACK}.parquet}

KENLM_ARGS=(); [ -n "$KENLM" ] && KENLM_ARGS=(--kenlm-path "$KENLM")

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python scripts/mms_zeroshot.py --track "$TRACK" --split "$SPLIT" --manifest "$MANIFEST" \
    --model-id "$MODEL" --target-lang "$LANGCODE" --run-name "$RUN" "${KENLM_ARGS[@]}"
