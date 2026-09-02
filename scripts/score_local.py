#!/usr/bin/env python
"""Score a predictions CSV with the OFFICIAL scorer (runtime/score.py), exactly as the leaderboard.

This shells out to `uv run runtime/score.py <gold> --predicted-path <pred>` so the number is
produced by the unmodified official script. For a quick in-process estimate (same normalize_text +
jiwer.wer) use lit.scoring.official.wer instead.

Usage:
    python scripts/score_local.py --gold data/manifests/id_jv_dev_gold.csv \
        --pred results/id_jv/<run>/submission.csv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCORE_PY = REPO_ROOT / "runtime" / "score.py"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="ground-truth CSV (audio_filename,transcript)")
    ap.add_argument("--pred", required=True, help="predictions CSV (audio_filename,transcript)")
    args = ap.parse_args()

    if not SCORE_PY.exists():
        sys.exit("runtime/score.py missing — run: git submodule update --init runtime")

    cmd = ["uv", "run", str(SCORE_PY), args.gold, "--predicted-path", args.pred]
    print("$", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
