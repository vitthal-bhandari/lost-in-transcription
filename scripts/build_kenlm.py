#!/usr/bin/env python
"""Build a KenLM from a manifest's training text, normalized to match our decode/scoring
convention (official normalize_text + fold_diacritics — same targets used for Qwen/Whisper FT).

Usage:
    python scripts/build_kenlm.py --manifest data/manifests/id_jv_plus_homostoria.parquet \
        --splits train,val --order 4 --out data/lm/id_jv_plus_homostoria
"""

from __future__ import annotations

import argparse

import pandas as pd

from lit.lm.kenlm_build import build_lm
from lit.scoring.official import normalize_text
from lit.text.normalize import fold_diacritics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", default="train,val", help="comma-separated split values to pull")
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.manifest)
    splits = set(args.splits.split(","))
    sub = df[df["split"].isin(splits)]
    print(f"[build_kenlm] {len(sub)} utterances from splits {sorted(splits)}")

    texts = [fold_diacritics(normalize_text(t)) for t in sub["text"].tolist()]
    build_lm(texts, args.out, order=args.order)


if __name__ == "__main__":
    main()
