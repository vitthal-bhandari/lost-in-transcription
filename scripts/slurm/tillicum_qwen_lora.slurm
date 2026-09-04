#!/bin/bash
#SBATCH --job-name=lit-qwenlora
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# LoRA fine-tune Qwen3-ASR-1.7B on id_jv (preserves the base prior -> less overfitting than the
# full FT that regressed to 0.3913). Reuses the existing qwen_ft JSONL/WAVs.
# Prereq (once): scripts/make_qwen_jsonl.py already produced data/jember_javanese/qwen_ft/*.jsonl
# Usage: sbatch --export=ALL,EPOCHS=3,LR=1e-4,R=16 scripts/slurm/tillicum_qwen_lora.slurm
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
source .venv-qwen/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="${LIT_CACHE_ROOT:-/gpfs/scrubbed/$USER/.cache}/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE"

python -c "import peft"     2>/dev/null || uv pip install -q peft
python -c "import datasets" 2>/dev/null || uv pip install -q datasets

# Fetch Qwen's SFT script, then patch in env-guarded LoRA.
SFT=scripts/vendor/qwen3_asr_sft.py
mkdir -p scripts/vendor
[ -f "$SFT" ] || curl -sL "https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/main/finetuning/qwen3_asr_sft.py" -o "$SFT"
python scripts/qwen_lora_patch.py "$SFT" scripts/vendor/qwen3_asr_sft_lora.py

FT_DIR=data/jember_javanese/qwen_ft
OUT=${OUT:-checkpoints/id_jv/qwen_lora}
EPOCHS=${EPOCHS:-3}; LR=${LR:-1e-4}; R=${R:-16}; ALPHA=${ALPHA:-32}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=== qwen LoRA FT: epochs=$EPOCHS lr=$LR r=$R alpha=$ALPHA -> $OUT ==="
QWEN_LORA=1 QWEN_LORA_R="$R" QWEN_LORA_ALPHA="$ALPHA" python scripts/vendor/qwen3_asr_sft_lora.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file "$FT_DIR/train.jsonl" --eval_file "$FT_DIR/val.jsonl" \
  --output_dir "$OUT" --lr "$LR" --epochs "$EPOCHS" --batch_size 32 --grad_acc 4 \
  --save_steps 200 --save_total_limit 3

# Merge the latest adapter into an inference-ready full model for eval.
CKPT=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "=== merging adapter $CKPT -> $OUT/merged ==="
python scripts/qwen_merge_lora.py Qwen/Qwen3-ASR-1.7B "$CKPT" "$OUT/merged"
echo "=== eval with: ==="
echo "sbatch --export=ALL,TRACK=id_jv,SPLIT=dev,MODEL=$OUT/merged,LANGCODE=Indonesian,RUN=qwen_lora scripts/slurm/tillicum_qwen.slurm"
