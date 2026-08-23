#!/usr/bin/env python
"""Run a baseline model on a track's local test/dev split and log WER.

Usage:
    python scripts/run_baseline.py --track es_en --split test --model whisper
Writes predictions + a metrics json under results/<track>/<run_name>/.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from lit.config import get_track
from lit.scoring import score_dataframe

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_transcriber(model: str, cfg: dict):
    if model == "whisper":
        from lit.models.whisper_infer import WhisperConfig, WhisperTranscriber
        return WhisperTranscriber(WhisperConfig(model_id=cfg["baseline_model"]))
    if model == "mms":
        from lit.models.mms_infer import MmsConfig, MmsTranscriber
        return MmsTranscriber(MmsConfig(model_id=cfg["baseline_model"]))
    if model == "omni":
        from lit.models.omni_infer import OMNI_LANG, OmniConfig, OmniTranscriber
        primary = cfg["langs"][0]
        return OmniTranscriber(OmniConfig(lang=OMNI_LANG.get(primary, "spa_Latn")))
    raise ValueError(f"unknown model '{model}'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="whisper", choices=["whisper", "mms", "omni"])
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    cfg = get_track(args.track)
    manifest_path = args.manifest or f"data/manifests/{args.track}.parquet"
    df = pd.read_parquet(manifest_path)
    df = df[df["split"] == args.split].reset_index(drop=True)

    transcriber = build_transcriber(args.model, cfg)
    df["prediction"] = transcriber.transcribe_batch(df["audio_path"].tolist())

    result = score_dataframe(df, profile=cfg["normalize"])
    run_name = args.run_name or f"{args.model}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    out_dir = REPO_ROOT / "results" / args.track / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "predictions.parquet")
    (out_dir / "metrics.json").write_text(
        json.dumps({"track": args.track, "split": args.split, "model": args.model,
                    "baseline_model": cfg["baseline_model"], **result.as_dict()}, indent=2)
    )
    print(f"[{args.track}] {args.model} {args.split} WER = {result.wer:.4f}  -> {out_dir}")


if __name__ == "__main__":
    main()
