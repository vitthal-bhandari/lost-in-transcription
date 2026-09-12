#!/usr/bin/env python
"""Build a manifest for a track and write speaker/session-disjoint (or k-fold) splits.

Usage:
    python scripts/create_splits.py --track es_en --out data/manifests/es_en.parquet
    python scripts/create_splits.py --track id_jv --out data/manifests/id_jv.parquet \
        --extra homostoria=data/homostoria --extra hari_minggoean=data/hari_minggoean
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lit.config import get_track
from lit.data.prepare import PREPARERS
from lit.data.splits import make_splits


def _parse_extra(pairs: list[str]) -> dict[str, str]:
    out = {}
    for p in pairs:
        name, _, path = p.partition("=")
        if not path:
            raise SystemExit(f"--extra must be name=path, got: {p!r}")
        out[name] = path
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=list(PREPARERS))
    ap.add_argument("--out", required=True, help="output manifest path (.parquet)")
    ap.add_argument("--extra", action="append", default=[],
                    help="extra corpus to pool in, name=path (id_jv only; repeatable)")
    args = ap.parse_args()

    cfg = get_track(args.track)
    extra = _parse_extra(args.extra)
    if extra:
        manifest = PREPARERS[args.track](cfg["data_dir"], extra_dirs=extra)
    else:
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
