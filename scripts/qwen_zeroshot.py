#!/usr/bin/env python
"""Zero-shot Qwen3-ASR benchmark on a manifest split (Tillicum, .venv-qwen). Submittable base.

Decodes with a Qwen3-ASR model, scores with the OFFICIAL normalization, writes predictions.csv +
metrics.json under results/<track>/<run>/. Sweep --language auto vs Indonesian for the code-switched
id_jv track (Javanese isn't a named Qwen language, so 'auto' lets the model detect/merge).

Examples (see scripts/slurm/tillicum_qwen.slurm):
    python scripts/qwen_zeroshot.py --track id_jv --split dev --language auto --run-name qwen17b_auto
    python scripts/qwen_zeroshot.py --track id_jv --split dev --language Indonesian --run-name qwen17b_ind
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from lit.data.audio import load_segment
from lit.models.qwen_infer import QwenConfig, QwenTranscriber
from lit.scoring.wer import wer_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_SR = 16000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--model-id", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--language", default="auto",
                    help="Qwen full language name (e.g. Indonesian), or 'auto'/'none' for detection")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--fold-diacritics", default=True, action=argparse.BooleanOptionalAction)
    args = ap.parse_args()

    lang = None if args.language in (None, "auto", "none") else args.language

    manifest = args.manifest or f"data/manifests/{args.track}.parquet"
    df = pd.read_parquet(manifest)
    df = df[df["split"] == args.split].reset_index(drop=True)

    cfg = QwenConfig(model_id=args.model_id, language=lang, batch_size=args.batch_size)
    transcriber = QwenTranscriber(cfg)
    print(f"[qwen] model={cfg.model_id} language={lang or 'auto'} n={len(df)} split={args.split}")

    arrays = [load_segment(r.audio_path, r.start, r.end, TARGET_SR) for r in df.itertuples()]
    preds: list[str] = []
    for i in range(0, len(arrays), args.batch_size):
        preds.extend(transcriber.transcribe_arrays(arrays[i : i + args.batch_size]))
        print(f"  transcribed {min(i + args.batch_size, len(arrays))}/{len(arrays)}", flush=True)

    if args.fold_diacritics:
        from lit.text.normalize import fold_diacritics
        preds = [fold_diacritics(p) for p in preds]
    df["prediction"] = preds

    result = wer_corpus(df["text"].tolist(), df["prediction"].tolist())
    run_name = args.run_name or f"qwen_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    out_dir = REPO_ROOT / "results" / args.track / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    df["audio_filename"] = df["audio_path"].map(lambda p: Path(p).name)
    df[["audio_filename", "prediction"]].rename(columns={"prediction": "transcript"}).to_csv(
        out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(
        {"track": args.track, "split": args.split, "model": cfg.model_id,
         "language": lang or "auto", **result.as_dict()}, indent=2))
    print(f"[{args.track}] {run_name} {args.split} WER = {result.wer:.4f}  -> {out_dir}")


if __name__ == "__main__":
    main()
