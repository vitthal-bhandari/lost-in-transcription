#!/usr/bin/env python
"""Zero-shot Omni ASR benchmark on a manifest split (Tillicum, .venv-omni). Ceiling reference.

Decodes with an Omni model card, scores with the OFFICIAL normalization, writes predictions.csv +
metrics.json under results/<track>/<run>/. LLM cards need a --lang; sweep ind_Latn vs jav_Latn for
the code-switched id_jv track. CTC cards ignore --lang.

Examples (see scripts/slurm/tillicum_omni.slurm):
    python scripts/omni_zeroshot.py --track id_jv --split dev \
        --model-card omniASR_LLM_7B_v2 --lang ind_Latn --run-name omni_llm7b_ind
    python scripts/omni_zeroshot.py --track id_jv --split dev \
        --model-card omniASR_CTC_7B_v2 --run-name omni_ctc7b
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from lit.data.audio import load_segment
from lit.models.omni_infer import OMNI_LANG, OmniConfig, OmniTranscriber
from lit.scoring.wer import wer_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_SR = 16000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--model-card", default="omniASR_LLM_7B_v2")
    ap.add_argument("--lang", default=None,
                    help="Omni lang code (e.g. ind_Latn / jav_Latn). LLM only; CTC ignores it. "
                         "'none' or 'auto' => pass None.")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    lang = None if args.lang in (None, "none", "auto") else args.lang

    manifest = args.manifest or f"data/manifests/{args.track}.parquet"
    df = pd.read_parquet(manifest)
    df = df[df["split"] == args.split].reset_index(drop=True)

    cfg = OmniConfig(model_card=args.model_card, lang=lang, batch_size=args.batch_size)
    transcriber = OmniTranscriber(cfg)
    print(f"[omni] card={cfg.model_card} lang={lang} ctc={transcriber.is_ctc} "
          f"n={len(df)} split={args.split}")

    arrays = [load_segment(r.audio_path, r.start, r.end, TARGET_SR) for r in df.itertuples()]
    df["prediction"] = transcriber.transcribe_arrays(arrays)

    result = wer_corpus(df["text"].tolist(), df["prediction"].tolist())
    run_name = args.run_name or f"omni_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    out_dir = REPO_ROOT / "results" / args.track / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    df["audio_filename"] = df["audio_path"].map(lambda p: Path(p).name)
    df[["audio_filename", "prediction"]].rename(columns={"prediction": "transcript"}).to_csv(
        out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(
        {"track": args.track, "split": args.split, "model_card": cfg.model_card,
         "lang": lang, "is_ctc": transcriber.is_ctc, **result.as_dict()}, indent=2))
    print(f"[{args.track}] {run_name} {args.split} WER = {result.wer:.4f}  -> {out_dir}")


if __name__ == "__main__":
    main()
