#!/bin/bash
#SBATCH --job-name=lit-eval
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Evaluate a Whisper model (zero-shot HF id OR a fine-tuned checkpoint dir) on a split.
# Usage:
#   sbatch --export=ALL,TRACK=id_jv,SPLIT=dev,MODEL=openai/whisper-large-v3,RUN=whisper_zeroshot \
#          scripts/slurm/tillicum_eval.slurm
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/gpfs/scrubbed/$USER/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE"

TRACK=${TRACK:?set TRACK}
SPLIT=${SPLIT:-dev}
MODEL=${MODEL:?set MODEL=<hf id or checkpoint dir>}
RUN=${RUN:-eval}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python scripts/eval_checkpoint.py --track "$TRACK" --split "$SPLIT" --model "$MODEL" --run-name "$RUN"
