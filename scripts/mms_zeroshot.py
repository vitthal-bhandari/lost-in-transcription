#!/usr/bin/env python
"""Zero-shot MMS-1B-all benchmark on a manifest split, with optional KenLM decoding.

Isolates the LM's contribution: run once with --kenlm-path unset (greedy CTC) and once with it
set (beam search + our in-domain LM) on the SAME acoustic model, to see how much the LM alone is
worth before any acoustic fine-tuning.

Examples (see scripts/slurm/tillicum_mms.slurm):
    python scripts/mms_zeroshot.py --track id_jv --split dev --run-name mms_greedy
    python scripts/mms_zeroshot.py --track id_jv --split dev \
        --kenlm-path data/lm/id_jv_plus_homostoria/lm_4gram.bin --run-name mms_kenlm
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from lit.data.audio import load_segment
from lit.models.mms_infer import MMS_LANG, MmsConfig, MmsTranscriber
from lit.scoring.wer import wer_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_SR = 16000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--model-id", default="facebook/mms-1b-all")
    ap.add_argument("--target-lang", default=None, help="MMS adapter code; default from track's matrix language")
    ap.add_argument("--kenlm-path", default=None, help="path to a KenLM .bin; omit for greedy decode")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.5)
    ap.add_argument("--beam-width", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--fold-diacritics", default=True, action=argparse.BooleanOptionalAction)
    args = ap.parse_args()

    lang = args.target_lang or MMS_LANG.get("id", "ind")

    manifest = args.manifest or f"data/manifests/{args.track}.parquet"
    df = pd.read_parquet(manifest)
    df = df[df["split"] == args.split].reset_index(drop=True)

    cfg = MmsConfig(model_id=args.model_id, target_lang=lang, kenlm_path=args.kenlm_path,
                    alpha=args.alpha, beta=args.beta, beam_width=args.beam_width,
                    batch_size=args.batch_size)
    transcriber = MmsTranscriber(cfg)
    print(f"[mms] model={cfg.model_id} lang={lang} kenlm={cfg.kenlm_path} n={len(df)} split={args.split}")

    arrays = [load_segment(r.audio_path, r.start, r.end, TARGET_SR) for r in df.itertuples()]
    preds = transcriber.transcribe_arrays(arrays)

    if args.fold_diacritics:
        from lit.text.normalize import fold_diacritics
        preds = [fold_diacritics(p) for p in preds]
    df["prediction"] = preds

    result = wer_corpus(df["text"].tolist(), df["prediction"].tolist())
    run_name = args.run_name or f"mms_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    out_dir = REPO_ROOT / "results" / args.track / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    df["audio_filename"] = df["audio_path"].map(lambda p: Path(p).name)
    df[["audio_filename", "prediction"]].rename(columns={"prediction": "transcript"}).to_csv(
        out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(
        {"track": args.track, "split": args.split, "model": cfg.model_id, "lang": lang,
         "kenlm_path": cfg.kenlm_path, "alpha": cfg.alpha, "beta": cfg.beta,
         **result.as_dict()}, indent=2))
    print(f"[{args.track}] {run_name} {args.split} WER = {result.wer:.4f}  -> {out_dir}")


if __name__ == "__main__":
    main()
