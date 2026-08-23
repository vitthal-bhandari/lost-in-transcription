#!/usr/bin/env python
"""Competition inference entry point (runs inside the submission container, OFFLINE).

Reads every audio file (.mp3/.wav) under $INPUT_DIR and writes $OUTPUT_CSV with columns
`audio_filename,transcript`. Must not touch the network or any hosted API. Model weights are
baked into the image under ./assets. Each test item is processed independently (no cross-item
leakage), per the rules.

STUB: select the per-track transcriber and loop. Confirm exact I/O paths + CSV schema against
the official submission spec before finalizing (login-gated).
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

AUDIO_EXTS = {".mp3", ".wav"}


def find_audio(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def load_transcriber():
    # TODO: instantiate the chosen model with weights from ./assets (baked into the image).
    raise NotImplementedError("submission.predict.load_transcriber: wire up the final model")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=os.environ.get("INPUT_DIR", "/data/test"))
    ap.add_argument("--output-csv", default=os.environ.get("OUTPUT_CSV", "/data/submission.csv"))
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    files = find_audio(input_dir)
    transcriber = load_transcriber()

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_filename", "transcript"])
        for path in files:
            transcript = transcriber.transcribe(str(path))
            writer.writerow([path.name, transcript])
    print(f"wrote {len(files)} predictions -> {args.output_csv}")


if __name__ == "__main__":
    main()
