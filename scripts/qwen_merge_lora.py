#!/usr/bin/env python
"""Merge a trained LoRA adapter into the base Qwen3-ASR model -> an inference-ready dir.

Our eval (Qwen3ASRModel.from_pretrained) loads a full model, not a PEFT adapter, so we fold the
adapter into the base weights and save the complete model + processor.

Usage: python scripts/qwen_merge_lora.py <base_id> <adapter_dir> <out_dir>
"""

import sys

import torch
from peft import PeftModel
from qwen_asr import Qwen3ASRModel


def main() -> None:
    base, adapter, out = sys.argv[1], sys.argv[2], sys.argv[3]
    w = Qwen3ASRModel.from_pretrained(base, dtype=torch.bfloat16, device_map=None)
    w.model = PeftModel.from_pretrained(w.model, adapter).merge_and_unload()
    w.model.save_pretrained(out)
    w.processor.save_pretrained(out)
    print("merged ->", out)


if __name__ == "__main__":
    main()
