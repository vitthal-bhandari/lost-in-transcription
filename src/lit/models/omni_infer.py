"""Meta Omnilingual ASR (omniASR) inference wrapper.

Why it matters here: Omni ASR natively covers ALL our tracks' languages — spa_Latn, eng_Latn,
ind_Latn, jav_Latn, and many Nahuatl varieties (nhi_Latn, nhw_Latn, azz_Latn, ...) — which
Whisper and MMS largely lack for Javanese/Nahuatl. Apache-2.0 weights + code, runs fully offline
from cached weights, so it satisfies the competition's open-weight / no-hosted-API rules.

Integration notes (see facebookresearch/omnilingual-asr):
  - `pip install omnilingual-asr` (fairseq2 backend, needs libsndfile). NOT a transformers model.
  - Weights cache to ~/.cache/fairseq2/assets/; pre-download then run offline (bake into the
    submission image).
  - Language is passed explicitly as `{code}_{script}`, e.g. "spa_Latn".
  - **40s audio limit** on standard variants; use an "Unlimited" variant OR segment audio (VAD)
    for long-form test items. Confirm test segment lengths against the data spec.
  - Variants: omniASR CTC (fast, ~2-15 GiB VRAM) and LLM 300M/1B/7B (7B ~17 GiB VRAM, ~30 GiB
    download). Pick by the submission GPU/runtime budget — 7B likely too heavy for the container.

STUB: wire up ASRInferencePipeline once the package is installed on the cluster.
"""

from __future__ import annotations

from dataclasses import dataclass

# Map our track language codes -> Omni's {code}_{script}. The es_nah entry is a PLACEHOLDER:
# confirm the exact Nahuatl variety of the W. Sierra Puebla corpus (candidates: nhi/nhw/azz).
OMNI_LANG = {
    "es": "spa_Latn",
    "en": "eng_Latn",
    "id": "ind_Latn",
    "jv": "jav_Latn",
    "nhi": "nhi_Latn",   # revisit: match the corpus's actual ISO 639-3 variety
}


@dataclass
class OmniConfig:
    model_card: str = "omniASR_CTC_1B"   # or omniASR_LLM_{300M,1B,7B}, *_Unlimited_* for long audio
    lang: str = "spa_Latn"               # primary decode language for the track
    batch_size: int = 2
    max_audio_sec: float = 40.0          # standard variants; segment above this
    device: str = "cuda"


class OmniTranscriber:
    def __init__(self, cfg: OmniConfig):
        self.cfg = cfg
        self._pipeline = None

    def _ensure_loaded(self):
        raise NotImplementedError(
            "OmniTranscriber: from omnilingual_asr.models.inference.pipeline import "
            "ASRInferencePipeline; init with cfg.model_card. Segment audio > cfg.max_audio_sec."
        )

    def transcribe(self, audio_path: str) -> str:
        self._ensure_loaded()
        raise NotImplementedError

    def transcribe_batch(self, audio_paths: list[str]) -> list[str]:
        self._ensure_loaded()
        raise NotImplementedError
