"""Whisper inference wrapper (zero-shot baseline + fine-tuned / LoRA checkpoints).

Decoding hygiene that tends to help on code-switched, conversational audio:
  - condition_on_previous_text=False  (stops hallucinated repetition drift)
  - temperature fallback              (recover from low-confidence decodes)
  - suppress non-speech tokens
  - optional initial_prompt

STUB: wire up transformers WhisperForConditionalGeneration / pipeline once deps are installed
and GPU is available. Kept import-light so the package imports without torch present.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WhisperConfig:
    model_id: str = "openai/whisper-large-v3"
    language: str | None = None          # None => let Whisper detect (better for code-switch)
    task: str = "transcribe"
    condition_on_previous_text: bool = False
    temperature: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    initial_prompt: str | None = None
    lora_adapter: str | None = None      # path to a PEFT adapter for a fine-tuned track
    device: str = "cuda"
    batch_size: int = 8
    extra: dict = field(default_factory=dict)


class WhisperTranscriber:
    def __init__(self, cfg: WhisperConfig):
        self.cfg = cfg
        self._model = None  # lazy-loaded

    def _ensure_loaded(self):
        raise NotImplementedError(
            "WhisperTranscriber: load model/processor here (transformers). "
            "Apply LoRA adapter if cfg.lora_adapter is set."
        )

    def transcribe(self, audio_path: str) -> str:
        self._ensure_loaded()
        raise NotImplementedError

    def transcribe_batch(self, audio_paths: list[str]) -> list[str]:
        self._ensure_loaded()
        raise NotImplementedError
