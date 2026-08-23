"""Common inference interface so baselines, fine-tuned models, and the submission bundle
all call transcription the same way."""

from __future__ import annotations

from typing import Protocol


class Transcriber(Protocol):
    """Anything that turns an audio file path into a transcript string."""

    def transcribe(self, audio_path: str) -> str: ...

    def transcribe_batch(self, audio_paths: list[str]) -> list[str]:
        return [self.transcribe(p) for p in audio_paths]
