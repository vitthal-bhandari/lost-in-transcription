"""Fine-tune Whisper on a track (full / LoRA / freeze-encoder), against a manifest.

Adapted from low-resource-asr/aft_whisper.py, with three changes for this competition:
  1. Data comes from our unified parquet manifest; train rows are utterance SEGMENTS of long-form
     recordings (sliced by start/end via lit.data.audio.load_segment). Dev/val are whole clips.
  2. Validation WER uses the OFFICIAL scorer normalization (lit.scoring.official) so the number
     tracks the leaderboard — this is metric_for_best_model.
  3. `strategy` selects full FT, LoRA (peft), or freeze-encoder, so one file covers the ablation.

Runs on Tillicum H200 (bf16). Hyperparameters are env-overridable (see TrainConfig). Trains on
split=="train", early-stops on split=="val"; evaluate the saved model on split=="dev" separately
(scripts/eval_checkpoint.py) — dev is the honest leaderboard proxy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from lit.data.audio import load_segment

TARGET_SR = 16000


def _env(name: str, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v is not None else default


@dataclass
class TrainConfig:
    model_id: str = field(default_factory=lambda: _env("WHISPER_MODEL", "openai/whisper-large-v3"))
    language: str = field(default_factory=lambda: _env("WHISPER_LANG", "id"))
    task: str = "transcribe"
    lr: float = field(default_factory=lambda: _env("LR", 1e-5, float))
    epochs: float = field(default_factory=lambda: _env("EPOCHS", 8, float))
    train_bs: int = field(default_factory=lambda: _env("TRAIN_BS", 16, int))
    eval_bs: int = field(default_factory=lambda: _env("EVAL_BS", 8, int))
    grad_accum: int = field(default_factory=lambda: _env("GRAD_ACCUM", 2, int))
    warmup_ratio: float = field(default_factory=lambda: _env("WARMUP_RATIO", 0.05, float))
    weight_decay: float = field(default_factory=lambda: _env("WEIGHT_DECAY", 0.0, float))
    gen_max_len: int = field(default_factory=lambda: _env("GEN_MAX_LEN", 225, int))
    patience: int = field(default_factory=lambda: _env("PATIENCE", 3, int))
    target_norm: str = field(default_factory=lambda: _env("TARGET_NORM", "official"))  # official|raw
    num_proc: int = field(default_factory=lambda: _env("NUM_PROC", 1, int))
    seed: int = field(default_factory=lambda: _env("SEED", 13, int))


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # Whisper prepends the decoder-start token itself; drop it from labels if present.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def _target_text(text: str, mode: str) -> str:
    # Fold pepet/taling diacritics so targets match the dev/test plain-vowel convention (the
    # Jember transcripts use diacritics the scorer counts as errors). See lit.text.fold_diacritics.
    from lit.text.normalize import fold_diacritics
    if mode == "official":
        from lit.scoring.official import normalize_text
        return fold_diacritics(normalize_text(text))
    return fold_diacritics("" if text is None else str(text))


def _build_dataset(df: pd.DataFrame, target_norm: str):
    import datasets

    d = df.copy()
    d["sentence"] = d["text"].map(lambda t: _target_text(t, target_norm))
    d = d[d["sentence"].str.strip() != ""]
    cols = ["audio_path", "start", "end", "sentence"]
    return datasets.Dataset.from_pandas(d[cols], preserve_index=False)


def _make_prepare_fn(processor: WhisperProcessor):
    max_label_length = getattr(processor.tokenizer, "model_max_length", 448)
    if not max_label_length or max_label_length > 1024:
        max_label_length = 448

    def prepare(batch: dict) -> dict:
        array = load_segment(batch["audio_path"], batch.get("start"), batch.get("end"), TARGET_SR)
        batch["input_features"] = processor.feature_extractor(
            array, sampling_rate=TARGET_SR
        ).input_features[0]
        batch["labels"] = processor.tokenizer(
            batch["sentence"], max_length=max_label_length, truncation=True
        ).input_ids
        return batch

    return prepare


def _make_compute_metrics(processor: WhisperProcessor):
    import jiwer

    from lit.scoring.official import normalize_text

    def compute_metrics(pred) -> dict:
        pred_ids, label_ids = pred.predictions, pred.label_ids
        label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        refs = [normalize_text(x) or "<empty>" for x in label_str]
        hyps = [normalize_text(x) for x in pred_str]
        return {"wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps)}

    return compute_metrics


def _apply_strategy(model, strategy: str):
    if strategy == "full":
        return model
    if strategy == "freeze_encoder":
        model.freeze_encoder()
        return model
    if strategy == "lora":
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(
            r=int(_env("LORA_R", 32, int)),
            lora_alpha=int(_env("LORA_ALPHA", 64, int)),
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
        return model
    raise ValueError(f"unknown strategy '{strategy}' (full|lora|freeze_encoder)")


def train(track_cfg: dict, manifest_path: str, out_dir: str, strategy: str = "full", **_) -> None:
    cfg = TrainConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)

    manifest = pd.read_parquet(manifest_path)
    train_df = manifest[manifest["split"] == "train"]
    val_df = manifest[manifest["split"] == "val"]
    print(f"[whisper/{strategy}] train={len(train_df)} val={len(val_df)} | model={cfg.model_id} "
          f"lang={cfg.language} lr={cfg.lr} epochs={cfg.epochs} target_norm={cfg.target_norm}")

    processor = WhisperProcessor.from_pretrained(cfg.model_id, language=cfg.language, task=cfg.task)
    processor.tokenizer.set_prefix_tokens(language=cfg.language, task=cfg.task)

    prepare = _make_prepare_fn(processor)
    train_ds = _build_dataset(train_df, cfg.target_norm).map(
        prepare, remove_columns=["audio_path", "start", "end", "sentence"], num_proc=cfg.num_proc)
    val_ds = _build_dataset(val_df, cfg.target_norm).map(
        prepare, remove_columns=["audio_path", "start", "end", "sentence"], num_proc=cfg.num_proc)

    model = WhisperForConditionalGeneration.from_pretrained(cfg.model_id)
    model.generation_config.language = cfg.language
    model.generation_config.task = cfg.task
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # incompatible with gradient checkpointing
    model = _apply_strategy(model, strategy)
    # Required for gradient checkpointing to backprop into inputs (esp. LoRA/frozen base).
    model.enable_input_require_grads()

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=model.config.decoder_start_token_id)

    args = Seq2SeqTrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=cfg.train_bs,
        per_device_eval_batch_size=cfg.eval_bs,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.epochs,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # fixes double-backward on step 0
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=cfg.gen_max_len,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        report_to=["none"],
        seed=cfg.seed,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=_make_compute_metrics(processor),
        processing_class=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.patience)],
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"[whisper/{strategy}] best val WER={metrics.get('eval_wer'):.4f} "
          f"CER={metrics.get('eval_cer'):.4f}")

    processor.save_pretrained(out)
    if strategy == "lora":
        model.save_pretrained(out)                      # adapter
        merged = model.merge_and_unload()
        merged.save_pretrained(out / "merged")          # ready-to-serve full model
        processor.save_pretrained(out / "merged")
    else:
        trainer.save_model(str(out))
    print(f"[whisper/{strategy}] saved -> {out}")
