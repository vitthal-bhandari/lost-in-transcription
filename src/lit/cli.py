"""Thin CLI dispatch: `python -m lit.cli train --track ... --approach ...`."""

from __future__ import annotations

import argparse

from lit.config import get_track

APPROACHES = {
    "whisper_lora": ("lit.training.aft_whisper", {"use_lora": True}),
    "whisper_full": ("lit.training.aft_whisper", {"use_lora": False}),
    "mms_ctc_lm": ("lit.training.aft_mms", {}),
    "xlsr_ctc_lm": ("lit.training.aft_xlsr", {}),
}


def _train(args: argparse.Namespace) -> None:
    import importlib

    cfg = get_track(args.track)
    approach = args.approach or cfg.get("approach", "whisper_lora")
    if approach not in APPROACHES:
        raise SystemExit(f"unknown approach '{approach}'; known: {list(APPROACHES)}")
    module_name, kwargs = APPROACHES[approach]
    module = importlib.import_module(module_name)
    manifest = args.manifest or f"data/manifests/{args.track}.parquet"
    module.train(cfg, manifest, args.out, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(prog="lit")
    sub = ap.add_subparsers(required=True)
    t = sub.add_parser("train", help="fine-tune a model on a track")
    t.add_argument("--track", required=True)
    t.add_argument("--approach", default=None, help="override configs/tracks.yaml approach")
    t.add_argument("--manifest", default=None)
    t.add_argument("--out", required=True)
    t.set_defaults(func=_train)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
