"""Audio loading with optional segment slicing, shared by all trainers/inference.

Train rows are utterance segments of long-form recordings (carry start/end seconds); dev/test rows
are whole clips (start/end = NaN). We load with soundfile first (fast, robust for mp3 via
libsndfile>=1.1, and avoids torchcodec/FFmpeg quirks on HPC) and fall back to librosa's
offset/duration path. Output is always mono float32 at target_sr.
"""

from __future__ import annotations

import math

import numpy as np


def _is_nan(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except TypeError:
        return False


def load_segment(audio_path: str, start=None, end=None, target_sr: int = 16000) -> np.ndarray:
    """Return a mono float32 waveform at target_sr for [start, end) seconds (or the whole file)."""
    import soundfile as sf

    seg = not (_is_nan(start) or _is_nan(end))
    try:
        if seg:
            info = sf.info(audio_path)
            sr = info.samplerate
            start_frame = max(0, int(float(start) * sr))
            stop_frame = min(info.frames, int(round(float(end) * sr)))
            array, sr = sf.read(audio_path, start=start_frame, stop=stop_frame, dtype="float32")
        else:
            array, sr = sf.read(audio_path, dtype="float32")
    except Exception:
        import librosa

        offset = float(start) if seg else 0.0
        duration = (float(end) - float(start)) if seg else None
        array, sr = librosa.load(audio_path, sr=None, mono=False, offset=offset, duration=duration)
        array = array.T if getattr(array, "ndim", 1) > 1 else array

    if getattr(array, "ndim", 1) > 1:            # stereo -> mono
        array = array.mean(axis=1)
    array = np.asarray(array, dtype=np.float32)

    if sr != target_sr:
        import librosa

        array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)
    return array
