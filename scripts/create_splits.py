#!/usr/bin/env python
"""Build a manifest for a track and write speaker/session-disjoint (or k-fold) splits.

Usage:
    python scripts/create_splits.py --track es_en --out data/manifests/es_en.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lit.config import get_track
from lit.data.prepare import PREPARERS
from lit.data.splits import make_splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=list(PREPARERS))
    ap.add_argument("--out", required=True, help="output manifest path (.parquet)")
    args = ap.parse_args()

    cfg = get_track(args.track)
    manifest = PREPARERS[args.track](cfg["data_dir"])
    manifest = make_splits(manifest, cfg["split"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(out)
    if "fold" in manifest.columns:
        print(manifest["fold"].value_counts().sort_index())
    else:
        print(manifest["split"].value_counts())
    print(f"wrote {len(manifest)} rows -> {out}")


if __name__ == "__main__":
    main()
