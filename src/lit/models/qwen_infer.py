"""Qwen3-ASR inference wrapper (open-weight, offline — SUBMITTABLE, it's in the runtime).

qwen-asr is a local model runner (not an API client): Qwen3-ASR-1.7B / 0.6B, 52 languages, runs
fully offline after weights download. Because `qwen-asr` + `vllm` are in the competition runtime,
a Qwen-based pipeline can be submitted directly (unlike Omni). Strong multilingual ASR — a candidate
new base if it beats fine-tuned Whisper (0.2589 dev) on the code-switched id_jv track.

API (transformers backend): Qwen3ASRModel.from_pretrained(id, ...).transcribe(audio=[(np,sr),...],
language=None|"Indonesian"). Language is a full NAME (not a code); None = auto-detect (can return a
merged label like "Indonesian,English" for code-switch). Imports are lazy so this module stays
importable in the main venv; run in the isolated .venv-qwen (scripts/slurm/setup_qwen_env.sh).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Track language -> Qwen full language name. Javanese may be unsupported by Qwen's 52 langs, so map
# to None (auto-detect) and let the model decide; Indonesian (the matrix language) is supported.
QWEN_LANG = {"id": "Indonesian", "en": "English", "es": "Spanish", "jv": None, "nhi": None}

TARGET_SR = 16000


@dataclass
class QwenConfig:
    model_id: str = "Qwen/Qwen3-ASR-1.7B"   # or Qwen/Qwen3-ASR-0.6B
    language: str | None = None              # None = auto-detect; or a full name e.g. "Indonesian"
    batch_size: int = 32
    max_new_tokens: int = 256
    device: str = "cuda:0"


class QwenTranscriber:
    def __init__(self, cfg: QwenConfig):
        self.cfg = cfg
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from qwen_asr import Qwen3ASRModel

        self._model = Qwen3ASRModel.from_pretrained(
            self.cfg.model_id,
            dtype=torch.bfloat16,
            device_map=self.cfg.device,
            max_inference_batch_size=self.cfg.batch_size,
            max_new_tokens=self.cfg.max_new_tokens,
        )

    def transcribe_arrays(self, arrays: list[np.ndarray]) -> list[str]:
        """Transcribe pre-decoded mono float32 16kHz waveforms."""
        self._ensure_loaded()
        audio = [(np.asarray(a, dtype=np.float32), TARGET_SR) for a in arrays]
        results = self._model.transcribe(audio=audio, language=self.cfg.language)
        return [r.text.strip() for r in results]
