#!/usr/bin/env python
"""Merge a trained LoRA adapter into the base Qwen3-ASR model -> an inference-ready dir.

Our eval (Qwen3ASRModel.from_pretrained) loads a full model, not a PEFT adapter, so we fold the
adapter into the base weights and save the complete model + processor.

Usage: python scripts/qwen_merge_lora.py <base_id> <adapter_dir> <out_dir>
"""

import glob
import os
import sys

import torch
from peft import PeftModel
from qwen_asr import Qwen3ASRModel

WEIGHT_GLOBS = ("*.safetensors", "pytorch_model.bin", "model.safetensors.index.json")


def main() -> None:
    base, adapter, out = sys.argv[1], sys.argv[2], sys.argv[3]
    print(f"[merge] loading base {base} ...", flush=True)
    w = Qwen3ASRModel.from_pretrained(base, dtype=torch.bfloat16, device_map=None)
    print(f"[merge] loading LoRA adapter from {adapter} ...", flush=True)
    w.model = PeftModel.from_pretrained(w.model, adapter).merge_and_unload()
    os.makedirs(out, exist_ok=True)
    print(f"[merge] saving merged model -> {out} ...", flush=True)
    w.model.save_pretrained(out, safe_serialization=True)
    w.processor.save_pretrained(out)

    found = [f for pat in WEIGHT_GLOBS for f in glob.glob(os.path.join(out, pat))]
    if not found:
        raise RuntimeError(
            f"merge produced NO weight files in {out} (checked {WEIGHT_GLOBS}); "
            "save_pretrained silently failed to write weights"
        )
    print(f"[merge] verified weight file(s): {found}")
    print("merged ->", out)


if __name__ == "__main__":
    main()
