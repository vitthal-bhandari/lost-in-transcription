"""Meta Omnilingual ASR (omniASR) inference wrapper — real API (fairseq2 pipeline).

LOCAL RESEARCH ONLY: fairseq2/omnilingual-asr are NOT in the competition runtime, so Omni output
cannot be submitted directly (pending a runtime dependency PR). Use for ceiling benchmarks and as a
teacher for pseudo-labeling/distillation into a submittable model. Runs on Tillicum in a dedicated
venv (scripts/slurm/setup_omni_env.sh); imports are lazy so this module is importable in the main
venv (which has no fairseq2).

Language conditioning (the thing to get right):
  - CTC models IGNORE `lang` (language-agnostic acoustic decode) — we pass None.
  - LLM models USE `lang` (one code per clip). For code-switched Indonesian-Javanese there is no
    single "correct" code, so we sweep ind_Latn vs jav_Latn on dev and let WER decide.
Codes are `{iso639-3}_{script}`; see omnilingual_asr...wav2vec2_llama.lang_ids.supported_langs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Track language -> Omni {code}_{script}. es_nah placeholder pending the corpus's Nahuatl variety.
OMNI_LANG = {
    "es": "spa_Latn",
    "en": "eng_Latn",
    "id": "ind_Latn",
    "jv": "jav_Latn",
    "nhi": "nhi_Latn",
}

MAX_AUDIO_SEC = 40.0
TARGET_SR = 16000


@dataclass
class OmniConfig:
    model_card: str = "omniASR_LLM_7B_v2"   # or omniASR_CTC_7B_v2, omniASR_LLM_Unlimited_7B_v2, ...
    lang: str | None = None                  # e.g. "ind_Latn"; ignored for CTC; None allowed
    batch_size: int = 2
    max_sec: float = MAX_AUDIO_SEC
    truncate_long: bool = True               # clamp >max_sec clips (standard cards cap at 40s)
    device: str | None = None


def validate_lang(lang: str | None) -> str | None:
    """Raise if `lang` is not a supported Omni language code (None is allowed)."""
    if lang is None:
        return None
    from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs

    supported = set(supported_langs)
    if lang not in supported:
        near = sorted(c for c in supported if c[:3] == lang[:3])[:8]
        raise ValueError(f"lang '{lang}' not supported by Omni. Nearby codes: {near or 'none'}")
    return lang


class OmniTranscriber:
    def __init__(self, cfg: OmniConfig):
        self.cfg = cfg
        self.is_ctc = "CTC" in cfg.model_card
        self._pipe = None
        if not self.is_ctc:
            validate_lang(cfg.lang)  # fail fast on a bad LLM lang code

    def _ensure_loaded(self):
        if self._pipe is not None:
            return
        import torch
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        device = self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = ASRInferencePipeline(
            model_card=self.cfg.model_card, device=device, dtype=torch.bfloat16
        )

    def transcribe_arrays(self, arrays: list[np.ndarray]) -> list[str]:
        """Transcribe pre-decoded mono float32 16kHz waveforms."""
        self._ensure_loaded()
        max_samples = int(self.cfg.max_sec * TARGET_SR)
        clipped = 0
        dicts = []
        for a in arrays:
            a = np.asarray(a, dtype=np.float32)
            if self.cfg.truncate_long and len(a) > max_samples:
                a = a[:max_samples]
                clipped += 1
            dicts.append({"waveform": a, "sample_rate": TARGET_SR})
        if clipped:
            print(f"[omni] truncated {clipped}/{len(arrays)} clips to {self.cfg.max_sec:.0f}s "
                  f"(use an omniASR_LLM_Unlimited_* card to avoid this)")

        # CTC ignores lang; LLM gets one code per clip.
        langs = None if self.is_ctc else [self.cfg.lang] * len(dicts)
        return self._pipe.transcribe(dicts, lang=langs, batch_size=self.cfg.batch_size)
