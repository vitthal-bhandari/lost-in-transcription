#!/usr/bin/env python
"""Materialize train/val segments to WAV and write Qwen3-ASR SFT JSONL.

Qwen's fine-tuner (qwen3_asr_sft.py) wants JSONL lines of {audio: <wav path>, text: <target>}.
Our train rows are utterance segments of long-form recordings, so we slice each to a 16kHz mono WAV
(via load_segment) and write the target as Qwen's prefixed format:
    "language Indonesian<asr_text>" + fold_diacritics(official_normalize(text))
matching (a) the zero-shot winner (force Indonesian) and (b) the dev/test plain-vowel convention.

Usage:
    python scripts/make_qwen_jsonl.py --track id_jv --language Indonesian --splits train,val \
        --out-dir data/jember_javanese/qwen_ft
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import soundfile as sf

from lit.data.audio import load_segment
from lit.scoring.official import normalize_text
from lit.text.normalize import fold_diacritics

TARGET_SR = 16000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="id_jv")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--language", default="Indonesian", help="Qwen language-prefix name")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--out-dir", default="data/jember_javanese/qwen_ft")
    args = ap.parse_args()

    manifest = args.manifest or f"data/manifests/{args.track}.parquet"
    df = pd.read_parquet(manifest)
    out_root = Path(args.out_dir)

    for split in args.splits.split(","):
        sub = df[df["split"] == split].reset_index(drop=True)
        wav_dir = out_root / split / "wavs"
        wav_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = out_root / f"{split}.jsonl"
        n = 0
        with open(jsonl_path, "w") as f:
            for i, row in enumerate(sub.itertuples()):
                target = fold_diacritics(normalize_text(row.text))
                if not target.strip():
                    continue
                arr = load_segment(row.audio_path, row.start, row.end, TARGET_SR)
                wav = wav_dir / f"{split}_{i:06d}.wav"
                sf.write(str(wav), arr, TARGET_SR)
                text = f"language {args.language}<asr_text>{target}"
                f.write(json.dumps({"audio": str(wav), "text": text}, ensure_ascii=False) + "\n")
                n += 1
                if n % 500 == 0:
                    print(f"  [{split}] wrote {n} ...", flush=True)
        print(f"[{split}] {n} examples -> {jsonl_path}")


if __name__ == "__main__":
    main()
