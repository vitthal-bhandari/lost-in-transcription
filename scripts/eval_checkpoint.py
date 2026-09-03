#!/usr/bin/env python
"""Evaluate a Whisper model (zero-shot HF id OR fine-tuned dir) on a manifest split.

Decodes every row, scores with the OFFICIAL normalization, and writes predictions.csv
(audio_filename,transcript) + metrics.json under results/<track>/<run_name>/. Run on Tillicum.

Examples:
    # zero-shot baseline on dev
    python scripts/eval_checkpoint.py --track id_jv --split dev \
        --model openai/whisper-large-v3 --run-name whisper_zeroshot
    # a fine-tuned checkpoint
    python scripts/eval_checkpoint.py --track id_jv --split dev \
        --model checkpoints/id_jv/whisper_full --run-name whisper_full
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from lit.config import get_track
from lit.data.audio import load_segment
from lit.scoring.wer import wer_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_SR = 16000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--model", required=True, help="HF model id or local checkpoint dir")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--language", default=None, help="override decode language (default: track lang)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--fold-diacritics", default=True, action=argparse.BooleanOptionalAction,
                    help="fold pepet/taling vowel diacritics to plain (id_jv convention); "
                         "--no-fold-diacritics to disable")
    args = ap.parse_args()

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    cfg = get_track(args.track)
    lang = args.language or cfg["langs"][0]
    manifest_path = args.manifest or f"data/manifests/{args.track}.parquet"
    df = pd.read_parquet(manifest_path)
    df = df[df["split"] == args.split].reset_index(drop=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(args.model, language=lang, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    preds: list[str] = []
    for i in range(0, len(df), args.batch_size):
        rows = df.iloc[i : i + args.batch_size]
        arrays = [load_segment(r.audio_path, r.start, r.end, TARGET_SR) for r in rows.itertuples()]
        inputs = processor(arrays, sampling_rate=TARGET_SR, return_tensors="pt")
        feats = inputs.input_features.to(device, dtype=dtype)
        with torch.no_grad():
            gen = model.generate(feats, language=lang, task="transcribe",
                                 condition_on_prev_tokens=False, max_new_tokens=225)
        preds.extend(t.strip() for t in processor.batch_decode(gen, skip_special_tokens=True))
        print(f"  decoded {min(i + args.batch_size, len(df))}/{len(df)}", flush=True)

    # Fold pepet/taling diacritics to the dev/test plain-vowel convention (see lit.text).
    if args.fold_diacritics:
        from lit.text.normalize import fold_diacritics
        preds = [fold_diacritics(p) for p in preds]
    df["prediction"] = preds
    result = wer_corpus(df["text"].tolist(), df["prediction"].tolist())

    run_name = args.run_name or f"whisper_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    out_dir = REPO_ROOT / "results" / args.track / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    df["audio_filename"] = df["audio_path"].map(lambda p: Path(p).name)
    df[["audio_filename", "prediction"]].rename(columns={"prediction": "transcript"}).to_csv(
        out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(
        {"track": args.track, "split": args.split, "model": args.model, "language": lang,
         **result.as_dict()}, indent=2))
    print(f"[{args.track}] {run_name} {args.split} WER = {result.wer:.4f}  -> {out_dir}")


if __name__ == "__main__":
    main()
