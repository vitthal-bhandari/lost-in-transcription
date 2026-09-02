#!/bin/bash
#SBATCH --job-name=lit-baseline
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8          # Tillicum fixed ratio: 8 CPU / 200G RAM per GPU
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-2                # one task (its own GPU) per track

# Zero-shot baseline WER on a track's dev split. MODEL in {whisper, mms, omni-local}.
# omni-local runs only on the cluster (research), never in the submission container.
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/gpfs/scrubbed/$USER/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE"

TRACKS=(es_en es_nah id_jv)
TRACK=${TRACKS[$SLURM_ARRAY_TASK_ID]}
MODEL=${MODEL:-whisper}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=== baseline: track=$TRACK model=$MODEL ==="
python scripts/run_baseline.py --track "$TRACK" --split dev --model "$MODEL"
