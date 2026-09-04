#!/bin/bash
#SBATCH --job-name=lit-qwen
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Zero-shot Qwen3-ASR benchmark on Tillicum H200 (isolated .venv-qwen).
# Usage:
#   sbatch --export=ALL,TRACK=id_jv,SPLIT=dev,MODEL=Qwen/Qwen3-ASR-1.7B,LANG=auto,RUN=qwen17b_auto \
#          scripts/slurm/tillicum_qwen.slurm
#   sbatch --export=ALL,TRACK=id_jv,LANG=Indonesian,RUN=qwen17b_ind scripts/slurm/tillicum_qwen.slurm
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
source .venv-qwen/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="${LIT_CACHE_ROOT:-/gpfs/scrubbed/$USER/.cache}/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE"

TRACK=${TRACK:?set TRACK}
SPLIT=${SPLIT:-dev}
MODEL=${MODEL:-Qwen/Qwen3-ASR-1.7B}
LANGCODE=${LANGCODE:-auto}          # 'auto' (detect) or a full name e.g. Indonesian
RUN=${RUN:-qwen}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python scripts/qwen_zeroshot.py --track "$TRACK" --split "$SPLIT" \
    --model-id "$MODEL" --language "$LANGCODE" --run-name "$RUN"
