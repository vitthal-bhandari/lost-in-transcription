#!/usr/bin/env bash
#SBATCH --job-name=lit-train
#SBATCH --partition=gpu-a40           # adjust to your Tillicum/Hyak partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"

# Usage: sbatch --export=TRACK=es_en,APPROACH=whisper_lora scripts/slurm/tillicum_train.slurm
TRACK=${TRACK:?set TRACK=es_en|es_nah|id_jv}
APPROACH=${APPROACH:-whisper_lora}
OUT_DIR=${OUT_DIR:-checkpoints/${TRACK}/${APPROACH}}

echo "=== train: track=$TRACK approach=$APPROACH out=$OUT_DIR ==="
python -m lit.cli train --track "$TRACK" --approach "$APPROACH" --out "$OUT_DIR"
