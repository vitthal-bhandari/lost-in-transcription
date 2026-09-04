#!/bin/bash
#SBATCH --job-name=lit-qwenft
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Fine-tune Qwen3-ASR-1.7B on id_jv using Qwen's official qwen3_asr_sft.py.
# Prereq (run once on login node, fast): build the JSONL + WAVs:
#   source .venv-qwen/bin/activate && PYTHONPATH=src \
#     python scripts/make_qwen_jsonl.py --track id_jv --language Indonesian --splits train,val
# Then: sbatch --export=ALL,EPOCHS=3,LR=2e-5 scripts/slurm/tillicum_qwen_ft.slurm
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
source .venv-qwen/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="${LIT_CACHE_ROOT:-/gpfs/scrubbed/$USER/.cache}/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE"

# datasets is required by the SFT script but not installed by setup_qwen_env.sh.
python -c "import datasets" 2>/dev/null || uv pip install -q datasets

# Vendor Qwen's official fine-tuning script (Tillicum has network for training).
SFT=scripts/vendor/qwen3_asr_sft.py
mkdir -p scripts/vendor
[ -f "$SFT" ] || curl -sL "https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/main/finetuning/qwen3_asr_sft.py" -o "$SFT"

FT_DIR=data/jember_javanese/qwen_ft
OUT=${OUT:-checkpoints/id_jv/qwen_ft}
EPOCHS=${EPOCHS:-3}; LR=${LR:-2e-5}; BS=${BS:-32}; GA=${GA:-4}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=== qwen FT: epochs=$EPOCHS lr=$LR bs=$BS grad_acc=$GA -> $OUT ==="
python "$SFT" \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file "$FT_DIR/train.jsonl" \
  --eval_file "$FT_DIR/val.jsonl" \
  --output_dir "$OUT" \
  --sr 16000 --batch_size "$BS" --grad_acc "$GA" --lr "$LR" --epochs "$EPOCHS" \
  --save_steps 200 --save_total_limit 3

# Qwen's SFT saves model+tokenizer but NOT the full processor (feature extractor + chat template),
# so Qwen3ASRModel.from_pretrained(checkpoint) fails. The processor is unchanged by fine-tuning,
# so save the complete base processor into each checkpoint (makes them self-contained).
python - "$OUT" <<'PY'
import glob, sys
import qwen_asr  # registers the Qwen3-ASR processor for AutoProcessor
from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained("Qwen/Qwen3-ASR-1.7B", fix_mistral_regex=True)
for ckpt in glob.glob(sys.argv[1] + "/checkpoint-*"):
    proc.save_pretrained(ckpt); print("processor ->", ckpt)
PY

echo "=== done. eval a checkpoint on dev with: ==="
echo "sbatch --export=ALL,TRACK=id_jv,SPLIT=dev,MODEL=$OUT/checkpoint-XXXX,LANG=Indonesian,RUN=qwen_ft scripts/slurm/tillicum_qwen.slurm"
