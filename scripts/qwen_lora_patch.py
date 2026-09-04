#!/usr/bin/env python
"""Inject env-guarded PEFT LoRA into Qwen's qwen3_asr_sft.py (it has no native LoRA).

Wraps the model with LoRA right after its generation_config is set (so patch_outer_forward has
already run on the raw module). Enabled at runtime via QWEN_LORA=1. Keeps the upstream script
otherwise untouched, so we can re-fetch/re-patch reproducibly.

Usage: python scripts/qwen_lora_patch.py <in.py> <out.py>
"""

import sys

ANCHOR = "    model.generation_config = GenerationConfig.from_model_config(model.config)\n"

INJECT = ANCHOR + '''
    import os as _os
    if _os.environ.get("QWEN_LORA", "0") == "1":
        from peft import LoraConfig, get_peft_model
        _cfg = LoraConfig(
            r=int(_os.environ.get("QWEN_LORA_R", "16")),
            lora_alpha=int(_os.environ.get("QWEN_LORA_ALPHA", "32")),
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, _cfg)
        model.print_trainable_parameters()
'''


def main() -> None:
    src = open(sys.argv[1]).read()
    if ANCHOR not in src:
        sys.exit("anchor not found — Qwen SFT script layout changed; update qwen_lora_patch.py")
    open(sys.argv[2], "w").write(src.replace(ANCHOR, INJECT, 1))
    print("wrote", sys.argv[2])


if __name__ == "__main__":
    main()
