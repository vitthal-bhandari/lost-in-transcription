"""MMS-1B-all CTC inference, with optional KenLM beam-search decoding via pyctcdecode.

MMS-1B-all is a single acoustic model with per-language ADAPTER weights (~1300 languages) that
must be explicitly selected — see facebook/mms-1b-all model card. We target Indonesian ("ind"),
the matrix language, matching the same choice that won for Qwen/Whisper.

KenLM integration follows the canonical HF pattern ("Boosting Wav2Vec2 with n-grams in
Transformers"): build a pyctcdecode decoder from the tokenizer's vocab + a KenLM binary, wrap in
Wav2Vec2ProcessorWithLM. Greedy (no-LM) mode is a plain CTC argmax decode for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TARGET_SR = 16000

# Track language -> MMS adapter code (ISO 639-3). Matches OMNI_LANG/QWEN_LANG conventions.
MMS_LANG = {"id": "ind", "en": "eng", "es": "spa", "jv": "jav", "nhi": "nhi"}


@dataclass
class MmsConfig:
    model_id: str = "facebook/mms-1b-all"
    target_lang: str = "ind"
    kenlm_path: str | None = None        # None = greedy CTC decode, no LM
    alpha: float = 0.5                    # LM weight
    beta: float = 1.5                     # word-insertion bonus
    beam_width: int = 100
    device: str = "cuda"
    batch_size: int = 8


class MmsTranscriber:
    def __init__(self, cfg: MmsConfig):
        self.cfg = cfg
        self._model = None
        self._processor = None          # plain Wav2Vec2Processor (greedy path)
        self._processor_lm = None       # Wav2Vec2ProcessorWithLM (KenLM path), if configured

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        cfg = self.cfg
        self._processor = AutoProcessor.from_pretrained(cfg.model_id, target_lang=cfg.target_lang)
        self._model = Wav2Vec2ForCTC.from_pretrained(
            cfg.model_id, target_lang=cfg.target_lang, ignore_mismatched_sizes=True
        )
        self._model.load_adapter(cfg.target_lang)
        self._model.to(cfg.device).eval()

        if cfg.kenlm_path:
            from pyctcdecode import build_ctcdecoder
            from transformers import Wav2Vec2ProcessorWithLM

            vocab_dict = self._processor.tokenizer.get_vocab()
            sorted_vocab = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda kv: kv[1])}
            decoder = build_ctcdecoder(
                labels=list(sorted_vocab.keys()),
                kenlm_model_path=cfg.kenlm_path,
                alpha=cfg.alpha,
                beta=cfg.beta,
            )
            self._processor_lm = Wav2Vec2ProcessorWithLM(
                feature_extractor=self._processor.feature_extractor,
                tokenizer=self._processor.tokenizer,
                decoder=decoder,
            )
            print(f"[mms] KenLM beam-search decoder ready: {cfg.kenlm_path}")
        else:
            print("[mms] greedy CTC decode (no KenLM)")

    def transcribe_arrays(self, arrays: list[np.ndarray]) -> list[str]:
        import torch

        self._ensure_loaded()
        cfg = self.cfg
        out: list[str] = []
        for i in range(0, len(arrays), cfg.batch_size):
            batch = [np.asarray(a, dtype=np.float32) for a in arrays[i : i + cfg.batch_size]]
            inputs = self._processor(batch, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = self._model(
                    inputs.input_values.to(cfg.device),
                    attention_mask=inputs.get("attention_mask", None).to(cfg.device)
                    if "attention_mask" in inputs else None,
                ).logits

            if self._processor_lm is not None:
                texts = self._processor_lm.batch_decode(
                    logits.cpu().numpy(), beam_width=cfg.beam_width
                ).text
            else:
                pred_ids = torch.argmax(logits, dim=-1)
                texts = self._processor.batch_decode(pred_ids)
            out.extend(t.strip() for t in texts)
        return out
