#!/usr/bin/env bash
#SBATCH --job-name=lit-baseline
#SBATCH --partition=gpu-a40           # adjust to your Tillicum/Hyak partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-2                    # one task per track

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"

TRACKS=(es_en es_nah id_jv)
TRACK=${TRACKS[$SLURM_ARRAY_TASK_ID]}
MODEL=${MODEL:-whisper}

echo "=== baseline: track=$TRACK model=$MODEL ==="
python scripts/run_baseline.py --track "$TRACK" --split test --model "$MODEL"
