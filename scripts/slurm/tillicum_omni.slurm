#!/bin/bash
#SBATCH --job-name=lit-omni
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Zero-shot Omni benchmark on Tillicum H200 (uses the isolated .venv-omni).
# Usage:
#   sbatch --export=ALL,TRACK=id_jv,SPLIT=dev,CARD=omniASR_LLM_7B_v2,LANG=ind_Latn,RUN=omni_llm7b_ind \
#          scripts/slurm/tillicum_omni.slurm
#   sbatch --export=ALL,TRACK=id_jv,SPLIT=dev,CARD=omniASR_CTC_7B_v2,RUN=omni_ctc7b \
#          scripts/slurm/tillicum_omni.slurm
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
source .venv-omni/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
# fairseq2 model assets on scratch (7B ~30GB); XDG_CACHE_HOME must match setup_omni_env.sh.
export XDG_CACHE_HOME="${LIT_CACHE_ROOT:-/gpfs/scrubbed/$USER/.cache}"
mkdir -p "$XDG_CACHE_HOME"

# fairseq2n dlopens the SYSTEM libsndfile.so.1, which Tillicum lacks. The soundfile wheel bundles
# libsndfile — expose it under a canonical name on the loader path.
SND=$(find "$SLURM_SUBMIT_DIR/.venv-omni/lib" -name 'libsndfile*.so*' 2>/dev/null | head -1)
if [ -n "$SND" ]; then
  mkdir -p "$SLURM_SUBMIT_DIR/.venv-omni/_snd"
  ln -sf "$SND" "$SLURM_SUBMIT_DIR/.venv-omni/_snd/libsndfile.so.1"
  export LD_LIBRARY_PATH="$SLURM_SUBMIT_DIR/.venv-omni/_snd:${LD_LIBRARY_PATH:-}"
fi

TRACK=${TRACK:?set TRACK}
SPLIT=${SPLIT:-dev}
CARD=${CARD:?set CARD=omniASR_LLM_7B_v2|omniASR_CTC_7B_v2|...}
LANGCODE=${LANGCODE:-none}          # LLM: ind_Latn / jav_Latn ; CTC: ignored
RUN=${RUN:-omni}
BATCH=${BATCH:-2}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python scripts/omni_zeroshot.py --track "$TRACK" --split "$SPLIT" \
    --model-card "$CARD" --lang "$LANGCODE" --batch-size "$BATCH" --run-name "$RUN"
