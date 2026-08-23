"""Local WER scorer — the single most important tool for avoiding wasted submissions.

WER = (S + D + I) / N over the whole corpus (competition definition), computed on text that
has passed through the track's normalization profile. Corpus-level (micro) WER is the default
because that is how leaderboards almost always aggregate; per-utterance mean is also reported
for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import jiwer
import pandas as pd

from lit.text.normalize import normalize


@dataclass
class WerResult:
    wer: float                 # corpus-level (S+D+I)/N
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_words: int
    n_utts: int
    utt_wer_mean: float        # mean of per-utterance WER (diagnostic only)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def wer_corpus(refs: list[str], hyps: list[str], profile: str = "default") -> WerResult:
    """Corpus WER over normalized refs/hyps. `profile` is a normalize.py profile name."""
    if len(refs) != len(hyps):
        raise ValueError(f"refs ({len(refs)}) and hyps ({len(hyps)}) length mismatch")

    norm_refs = [normalize(r, profile) for r in refs]
    norm_hyps = [normalize(h, profile) for h in hyps]

    # jiwer needs non-empty references; guard empties so alignment stays defined.
    safe_refs, safe_hyps = [], []
    for r, h in zip(norm_refs, norm_hyps):
        safe_refs.append(r if r.strip() else "<empty>")
        safe_hyps.append(h)

    out = jiwer.process_words(safe_refs, safe_hyps)
    ref_words = out.substitutions + out.deletions + out.hits
    corpus_wer = (
        (out.substitutions + out.deletions + out.insertions) / ref_words
        if ref_words else 0.0
    )
    per_utt = [
        jiwer.wer(r, h) for r, h in zip(safe_refs, safe_hyps)
    ]
    return WerResult(
        wer=corpus_wer,
        substitutions=out.substitutions,
        deletions=out.deletions,
        insertions=out.insertions,
        hits=out.hits,
        ref_words=ref_words,
        n_utts=len(safe_refs),
        utt_wer_mean=sum(per_utt) / len(per_utt) if per_utt else 0.0,
    )


def score_dataframe(
    df: pd.DataFrame,
    profile: str = "default",
    ref_col: str = "text",
    hyp_col: str = "prediction",
) -> WerResult:
    """Score a dataframe with reference and prediction columns."""
    return wer_corpus(df[ref_col].tolist(), df[hyp_col].tolist(), profile=profile)
