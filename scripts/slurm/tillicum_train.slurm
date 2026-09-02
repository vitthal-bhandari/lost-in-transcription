#!/bin/bash
#SBATCH --job-name=lit-train
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8          # Tillicum fixed ratio: 8 CPU / 200G RAM per GPU
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:1          # H200 = 141GB VRAM; CPU-only jobs are not allowed
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Fine-tune a model on a track using Tillicum H200 nodes.
# Usage:
#   sbatch --export=ALL,TRACK=id_jv,APPROACH=whisper_lora scripts/slurm/tillicum_train.slurm
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
# Keep model/dataset caches off the 10GB home quota, on scratch.
export HF_HOME="${HF_HOME:-/gpfs/scrubbed/$USER/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="${TORCH_HOME:-/gpfs/scrubbed/$USER/.cache/torch}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME"

TRACK=${TRACK:?set TRACK=id_jv|es_en|es_nah}
APPROACH=${APPROACH:-whisper_lora}
OUT_DIR=${OUT_DIR:-checkpoints/${TRACK}/${APPROACH}}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=== train: track=$TRACK approach=$APPROACH out=$OUT_DIR on $(hostname) ==="
python -m lit.cli train --track "$TRACK" --approach "$APPROACH" --out "$OUT_DIR"
