#!/usr/bin/env python
"""Emit a ground-truth CSV (audio_filename,transcript) from a manifest split, for official scoring.

Only meaningful for clip-based splits with unique per-row audio filenames (e.g. the official `dev`
split, or a test set). Segment-based train/val rows share a recording filename, so those are scored
in-process during training via lit.scoring.wer_corpus, not through the filename-join scorer.

Usage:
    python scripts/make_gold_csv.py --track id_jv --split dev --out data/manifests/id_jv_dev_gold.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = args.manifest or f"data/manifests/{args.track}.parquet"
    df = pd.read_parquet(manifest)
    df = df[df["split"] == args.split].copy()
    df["audio_filename"] = df["audio_path"].map(lambda p: Path(p).name)

    dups = df["audio_filename"].duplicated().sum()
    if dups:
        raise SystemExit(
            f"{dups} duplicate audio_filename in split '{args.split}' — this split is segment-based "
            "and can't be scored via the filename-join scorer. Score it in-process instead."
        )

    out = df[["audio_filename", "transcript"]] if "transcript" in df.columns else \
        df[["audio_filename", "text"]].rename(columns={"text": "transcript"})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} gold rows -> {args.out}")


if __name__ == "__main__":
    main()
