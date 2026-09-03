#!/usr/bin/env python
"""Error analysis for a predictions run: worst utterances + most frequent substitution pairs.

Joins results/<track>/<run>/predictions.csv (audio_filename,transcript=pred) with the manifest
split's references, normalizes both with the OFFICIAL scorer, then reports per-utterance WER and
the top substituted word pairs (surfaces e.g. Javanese->Indonesian swaps). CPU-only; run anywhere.

Usage:
    python scripts/error_analysis.py --track id_jv --split dev --run whisper_zeroshot --top 15
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import jiwer
import pandas as pd

from lit.scoring.official import normalize_text

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--run", required=True, help="results/<track>/<run>/ dir name")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    run_dir = REPO_ROOT / "results" / args.track / args.run
    pred = pd.read_csv(run_dir / "predictions.csv").rename(columns={"transcript": "pred"})
    manifest = args.manifest or f"data/manifests/{args.track}.parquet"
    mf = pd.read_parquet(manifest)
    mf = mf[mf["split"] == args.split].copy()
    mf["audio_filename"] = mf["audio_path"].map(lambda p: Path(p).name)
    df = mf[["audio_filename", "text"]].merge(pred, on="audio_filename", how="inner")

    df["ref_n"] = df["text"].map(normalize_text)
    df["hyp_n"] = df["pred"].map(normalize_text)
    df["uwer"] = [jiwer.wer(r or "<e>", h) for r, h in zip(df["ref_n"], df["hyp_n"])]

    print(f"\n=== {args.track}/{args.run} [{args.split}] — {len(df)} utts ===")
    print(f"corpus WER = {jiwer.wer(df['ref_n'].tolist(), df['hyp_n'].tolist()):.4f}\n")

    print(f"--- {args.top} worst utterances (by per-utt WER) ---")
    for r in df.sort_values("uwer", ascending=False).head(args.top).itertuples():
        print(f"[{r.uwer:.2f}] {r.audio_filename}")
        print(f"   REF: {r.ref_n[:160]}")
        print(f"   HYP: {r.hyp_n[:160]}")

    # Most frequent substitution word-pairs across the corpus.
    out = jiwer.process_words(df["ref_n"].tolist(), df["hyp_n"].tolist())
    subs: Counter = Counter()
    for refs, hyps, chunks in zip(out.references, out.hypotheses, out.alignments):
        for c in chunks:
            if c.type == "substitute":
                for ri, hi in zip(range(c.ref_start_idx, c.ref_end_idx),
                                  range(c.hyp_start_idx, c.hyp_end_idx)):
                    subs[(refs[ri], hyps[hi])] += 1
    print(f"\n--- top {args.top} substitution pairs (ref -> hyp) ---")
    for (rw, hw), n in subs.most_common(args.top):
        print(f"  {n:4d}  {rw!r} -> {hw!r}")


if __name__ == "__main__":
    main()
