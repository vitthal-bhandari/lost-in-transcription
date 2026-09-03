#!/usr/bin/env python
"""Tabulate all results/<track>/*/metrics.json into one WER-sorted leaderboard.

Handles both Whisper runs (key 'model') and Omni runs (key 'model_card').

Usage:
    python scripts/leaderboard.py --track id_jv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="id_jv")
    args = ap.parse_args()

    rows = []
    for mj in sorted((REPO_ROOT / "results" / args.track).glob("*/metrics.json")):
        d = json.loads(mj.read_text())
        rows.append({
            "run": mj.parent.name,
            "split": d.get("split", "?"),
            "wer": d.get("wer", float("nan")),
            "model": d.get("model") or d.get("model_card", "?"),
            "S": d.get("substitutions", 0),
            "D": d.get("deletions", 0),
            "I": d.get("insertions", 0),
            "n": d.get("n_utts", 0),
        })
    if not rows:
        print(f"no results under results/{args.track}/")
        return

    rows.sort(key=lambda r: (r["split"], r["wer"]))
    w = max(len(r["run"]) for r in rows)
    print(f"\n{'run':<{w}}  {'split':<5}  {'WER':>7}   {'S':>5} {'D':>5} {'I':>5}  {'n':>4}  model")
    print("-" * (w + 60))
    for r in rows:
        print(f"{r['run']:<{w}}  {r['split']:<5}  {r['wer']:>7.4f}   "
              f"{r['S']:>5} {r['D']:>5} {r['I']:>5}  {r['n']:>4}  {r['model']}")
    print()


if __name__ == "__main__":
    main()
