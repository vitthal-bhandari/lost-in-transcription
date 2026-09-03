"""Competition submission entry point (packed to submission.zip as `main.py`).

Runs INSIDE the official runtime container (Python 3.12, no network) via `uv run src/main.py`.
Contract (see runtime/ submodule, examples/template/main.py):
  - read   /code_execution/data/submission_format.csv  (cols: audio_filename, transcript)
  - read   /code_execution/data/clips/<audio_filename>
  - write  /code_execution/submission/submission.csv    (cols: audio_filename, transcript)

Decoding mirrors scripts/eval_checkpoint.py EXACTLY (soundfile decode + WhisperProcessor +
model.generate) so the submission reproduces our measured dev WER — and, importantly, does NOT
shell out to the ffmpeg binary (the transformers pipeline path-input mode does, which is fragile).
Model weights are baked into the zip under ./assets/model (no network in the container).

Local testing: override paths with env vars, e.g.
    DATA_DIR=data/jember_javanese/indonesian_dev \
    SUBMISSION_FORMAT_CSV=... CLIPS_DIR=... OUTPUT_CSV=/tmp/sub.csv \
    MODEL_DIR=checkpoints/id_jv/whisper_full  python submission_src/main.py
"""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path

import numpy as np
import polars as pl

DATA_DIR = Path(os.environ.get("DATA_DIR", "/code_execution/data"))
SUBMISSION_FORMAT_CSV = Path(os.environ.get("SUBMISSION_FORMAT_CSV", DATA_DIR / "submission_format.csv"))
CLIPS_DIR = Path(os.environ.get("CLIPS_DIR", DATA_DIR / "clips"))
OUTPUT_CSV = Path(os.environ.get("OUTPUT_CSV", "/code_execution/submission/submission.csv"))
MODEL_DIR = os.environ.get("MODEL_DIR", str(Path(__file__).parent / "assets" / "model"))

TARGET_SR = 16000
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
LANGUAGE = os.environ.get("WHISPER_LANG", "id")
# Diacritic folding aligns output with the dev/test plain-vowel convention (id_jv). Evidence: dev
# uses diacritics 8.9% vs train 69.2%. Net +WER on dev. Toggle OFF (FOLD_DIACRITICS=0) to A/B on the
# leaderboard if the public score suggests test uses a different convention.
FOLD_DIACRITICS = os.environ.get("FOLD_DIACRITICS", "1") != "0"


def _fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def load_audio(path: Path) -> np.ndarray:
    """Decode to mono float32 at TARGET_SR via soundfile (mp3 through libsndfile); no ffmpeg."""
    import soundfile as sf

    array, sr = sf.read(str(path), dtype="float32")
    if getattr(array, "ndim", 1) > 1:
        array = array.mean(axis=1)
    array = np.asarray(array, dtype=np.float32)
    if sr != TARGET_SR:
        import librosa

        array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
    return array


def main() -> None:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    submission = pl.read_csv(SUBMISSION_FORMAT_CSV)
    filenames = submission["audio_filename"].to_list()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(MODEL_DIR, language=LANGUAGE, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR, torch_dtype=dtype).to(device)
    model.eval()

    transcripts: list[str] = []
    for i in range(0, len(filenames), BATCH_SIZE):
        batch = filenames[i : i + BATCH_SIZE]
        arrays = [load_audio(CLIPS_DIR / f) for f in batch]
        inputs = processor(arrays, sampling_rate=TARGET_SR, return_tensors="pt")
        feats = inputs.input_features.to(device, dtype=dtype)
        with torch.no_grad():
            gen = model.generate(feats, language=LANGUAGE, task="transcribe",
                                 condition_on_prev_tokens=False, max_new_tokens=225)
        transcripts.extend(t.strip() for t in processor.batch_decode(gen, skip_special_tokens=True))
        print(f"  transcribed {min(i + BATCH_SIZE, len(filenames))}/{len(filenames)}", flush=True)

    if FOLD_DIACRITICS:
        transcripts = [_fold(t) for t in transcripts]

    submission = submission.with_columns(pl.Series("transcript", transcripts))
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    submission.write_csv(OUTPUT_CSV)
    print(f"Wrote {submission.height} predictions to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
