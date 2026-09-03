#!/bin/bash
#SBATCH --job-name=lit-smoke
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# End-to-end smoke test of the SUBMISSION pipeline (submission_src/main.py) on the dev clips:
# runs exactly the code that will run in the competition container, then scores the output with the
# official scorer. Confirms the submission reproduces our measured dev WER (~0.2592 for whisper_full)
# BEFORE we pack the zip / submit.
#
# Usage:
#   sbatch --export=ALL,MODEL_DIR=checkpoints/id_jv/whisper_full scripts/slurm/tillicum_smoke.slurm
#   # A/B the fold: add FOLD_DIACRITICS=0
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/gpfs/scrubbed/$USER/.cache/huggingface}"

MODEL_DIR=${MODEL_DIR:?set MODEL_DIR=checkpoints/id_jv/whisper_full}
OUT=results/id_jv/smoke_$(basename "$MODEL_DIR")
mkdir -p "$OUT"

# Dev gold doubles as the submission_format input (main.py only reads audio_filename).
python scripts/make_gold_csv.py --track id_jv --split dev --out "$OUT/dev_gold.csv"

# Run the real submission entry point against dev clips.
SUBMISSION_FORMAT_CSV="$OUT/dev_gold.csv" \
CLIPS_DIR=data/jember_javanese/indonesian_dev/clips \
OUTPUT_CSV="$OUT/submission.csv" \
MODEL_DIR="$MODEL_DIR" \
FOLD_DIACRITICS="${FOLD_DIACRITICS:-1}" \
  python submission_src/main.py

echo "=== official score of the submission output ==="
python scripts/score_local.py --gold "$OUT/dev_gold.csv" --pred "$OUT/submission.csv"
