"""Competition submission entry point (packed to submission.zip as `main.py`).

Runs INSIDE the official runtime container (Python 3.12, no network) via `uv run src/main.py`.
Contract (see runtime/ submodule, examples/template/main.py):
  - read   /code_execution/data/submission_format.csv  (cols: audio_filename, transcript)
  - read   /code_execution/data/clips/<audio_filename>
  - write  /code_execution/submission/submission.csv    (cols: audio_filename, transcript)

Model weights MUST be baked into the zip under ./assets/ (no network in the container). Set
MODEL_DIR to the packed model directory. Deps (torch/transformers/...) come from the runtime env.

Local testing: override paths with env vars, e.g.
    DATA_DIR=data/jember_javanese/indonesian_dev \
    SUBMISSION_FORMAT_CSV=... CLIPS_DIR=... OUTPUT_CSV=/tmp/sub.csv \
    MODEL_DIR=... uv run submission_src/main.py
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

DATA_DIR = Path(os.environ.get("DATA_DIR", "/code_execution/data"))
SUBMISSION_FORMAT_CSV = Path(os.environ.get("SUBMISSION_FORMAT_CSV", DATA_DIR / "submission_format.csv"))
CLIPS_DIR = Path(os.environ.get("CLIPS_DIR", DATA_DIR / "clips"))
OUTPUT_CSV = Path(os.environ.get("OUTPUT_CSV", "/code_execution/submission/submission.csv"))
# Packed model directory (baked into the zip). Filled once we have a fine-tuned checkpoint.
MODEL_DIR = os.environ.get("MODEL_DIR", str(Path(__file__).parent / "assets" / "model"))

TARGET_SR = 16000
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))


def _build_asr():
    """Load a transformers ASR pipeline from the packed MODEL_DIR (Whisper or CTC/MMS)."""
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "automatic-speech-recognition",
        model=MODEL_DIR,
        device=device,
        torch_dtype=torch.float16 if device == 0 else torch.float32,
        chunk_length_s=30,
    )


def main() -> None:
    submission = pl.read_csv(SUBMISSION_FORMAT_CSV)
    filenames = submission["audio_filename"].to_list()
    paths = [str(CLIPS_DIR / f) for f in filenames]

    asr = _build_asr()
    # Whisper decoding hygiene helps on conversational, code-switched audio.
    generate_kwargs = {"condition_on_prev_tokens": False, "temperature": 0.0}
    outputs = asr(paths, batch_size=BATCH_SIZE, generate_kwargs=generate_kwargs)
    transcripts = [o["text"].strip() for o in outputs]

    submission = submission.with_columns(pl.Series("transcript", transcripts))
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    submission.write_csv(OUTPUT_CSV)
    print(f"Wrote {submission.height} predictions to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
