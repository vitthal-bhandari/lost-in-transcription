"""Local WER scorer — uses the OFFICIAL normalization (src/lit/scoring/official.py) so local
numbers track the leaderboard. Adds S/D/I diagnostics on top of the official metric.

The headline `wer` equals `jiwer.wer(official_norm(refs), official_norm(hyps))`, which is exactly
what runtime/score.py computes. The substitution/deletion/insertion breakdown is extra detail for
error analysis and is computed on the same officially-normalized text.
"""

from __future__ import annotations

from dataclasses import dataclass

import jiwer
import pandas as pd

from .official import normalize_text


@dataclass
class WerResult:
    wer: float                 # corpus-level (S+D+I)/N — matches the official scorer
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_words: int
    n_utts: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def wer_corpus(refs: list[str], hyps: list[str]) -> WerResult:
    if len(refs) != len(hyps):
        raise ValueError(f"refs ({len(refs)}) and hyps ({len(hyps)}) length mismatch")

    norm_refs = [normalize_text(r) for r in refs]
    norm_hyps = [normalize_text(h) for h in hyps]
    # jiwer needs non-empty references to keep alignment defined.
    safe_refs = [r if r.strip() else "<empty>" for r in norm_refs]

    out = jiwer.process_words(safe_refs, norm_hyps)
    ref_words = out.substitutions + out.deletions + out.hits
    corpus_wer = (
        (out.substitutions + out.deletions + out.insertions) / ref_words if ref_words else 0.0
    )
    return WerResult(
        wer=corpus_wer,
        substitutions=out.substitutions,
        deletions=out.deletions,
        insertions=out.insertions,
        hits=out.hits,
        ref_words=ref_words,
        n_utts=len(safe_refs),
    )


def score_dataframe(
    df: pd.DataFrame,
    ref_col: str = "text",
    hyp_col: str = "prediction",
    **_ignored,  # tolerate a legacy `profile=` kwarg; official normalization is global now
) -> WerResult:
    return wer_corpus(df[ref_col].tolist(), df[hyp_col].tolist())
