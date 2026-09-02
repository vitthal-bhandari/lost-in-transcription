"""Build bilingual KenLM n-gram LMs from normalized transcripts (+ external text with rights).

A code-switch-aware LM = train text pooled across BOTH languages of a track so the decoder is
not penalized for switching mid-utterance. Mirrors the arpa_artifacts / lm_ablation workflow
from low-resource-asr.

STUB: shell out to KenLM's lmplz/build_binary; write lm_train.txt, lm_{3,4}gram.arpa per track.
"""

from __future__ import annotations

from pathlib import Path


def build_lm(train_texts: list[str], out_dir: str | Path, order: int = 4) -> Path:
    raise NotImplementedError(
        "build_lm: write normalized corpus to lm_train.txt, run `lmplz -o {order}`, "
        "then build_binary; return path to the .arpa/.bin"
    )
