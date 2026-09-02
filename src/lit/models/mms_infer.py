"""MMS / XLS-R CTC inference wrapper, with optional KenLM beam-search decoding.

For low-resource + code-switched tracks (es_nah, id_jv) a CTC acoustic model + bilingual n-gram
LM via pyctcdecode is often the strongest, most compute-frugal option — and mirrors the n-gram
ablation harness from the prior low-resource-asr project.

STUB: wire up Wav2Vec2ForCTC + Wav2Vec2ProcessorWithLM once deps/GPU are available.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MmsConfig:
    model_id: str = "facebook/mms-1b-all"
    target_lang: str | None = None       # MMS adapter language code, if used
    kenlm_path: str | None = None        # path to .arpa / .bin for beam-search rescoring
    alpha: float = 0.5                    # LM weight
    beta: float = 1.5                     # word-insertion bonus
    beam_width: int = 100
    device: str = "cuda"
    batch_size: int = 8


class MmsTranscriber:
    def __init__(self, cfg: MmsConfig):
        self.cfg = cfg
        self._model = None

    def _ensure_loaded(self):
        raise NotImplementedError(
            "MmsTranscriber: load Wav2Vec2ForCTC + processor; build pyctcdecode decoder "
            "from cfg.kenlm_path if provided."
        )

    def transcribe(self, audio_path: str) -> str:
        self._ensure_loaded()
        raise NotImplementedError

    def transcribe_batch(self, audio_paths: list[str]) -> list[str]:
        self._ensure_loaded()
        raise NotImplementedError
