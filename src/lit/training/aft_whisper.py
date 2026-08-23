"""Fine-tune Whisper (full or LoRA) on a track. Mirrors low-resource-asr/aft_whisper.py.

Primary approach for es_en (data-rich). For low-resource tracks prefer LoRA + augmentation.
STUB: implement Seq2SeqTrainer loop; normalize targets with the track's profile at data-prep
time so training and scoring agree.
"""

from __future__ import annotations


def train(track_cfg: dict, manifest_path: str, out_dir: str, use_lora: bool = True) -> None:
    raise NotImplementedError("aft_whisper.train: implement Seq2SeqTrainer / PEFT loop")
