"""Text normalization for training targets and WER scoring.

Adapted from the `clean_transcript` used in low-resource-asr (src/training/aft_mms.py):
that pipeline removes bracketed annotations, unintelligible markers, and most punctuation,
preserves ellipses / apostrophes / hyphens, and lowercases only at WER-compute time. We keep
that behavior faithfully and add:
  - a `nahuatl` profile that applies the (u/k/s) orthography conventions, and
  - lowercasing folded into the scoring profiles (matching how aft_mms computes WER: lower both
    sides before jiwer).

This is our interim scorer-parity layer. Replace with the official normalization the moment we
obtain the competition's scoring script (see EXPERIMENT_PLAN.md). `clean_transcript` is exposed
separately so training data-prep can reuse the exact same cleaning without forcing lowercase.
"""

from __future__ import annotations

import re

from .nahuatl_ortho import to_uks_orthography

# --- Regex patterns (ported verbatim from low-resource-asr aft_mms.py) ---
BRACKETED = re.compile(r"\[[^\]]+\]")        # [/], [//], [: gloss], [x 2], [= comment], @tags in []
UNINTELL_PAREN = re.compile(r"\(\?+\)")      # (?), (??)  unintelligible markers
REPL_PUNC = re.compile('[,?¿¡!";:]+')        # punctuation removed outright
MULTISPACE = re.compile("  +")

# Extra CHAT/CLAN codes not covered by the original repo but present in Bangor Miami:
#   &=laughs / &-uh / &+word  (event & fragment codes), @s:eng language tags, xxx/yyy/www,
#   0word omitted-word markers, <...> retracing scopes.
_CHAT_EXTRA = re.compile(r"&[=\-+]?\w+|@\S+|<[^>]*>|\bxxx\b|\byyy\b|\bwww\b|\b0\w+")
# CHAT pause/comment groups: (.), (..), (...), (0.5), (some comment). The original repo only
# stripped (?) via UNINTELL_PAREN; CHAT corpora use parentheses far more broadly.
_PAREN_GROUP = re.compile(r"\([^)]*\)")


def clean_transcript(text: str, *, drop_chat_extra: bool = True) -> str:
    """Clean a transcript (NO lowercasing), faithful to low-resource-asr's clean_transcript.

    `drop_chat_extra` also strips CHAT event/language/unintelligible codes common in the Bangor
    Miami corpus. Set False to match the original repo exactly.
    """
    if text is None:
        return ""
    text = str(text)
    text = BRACKETED.sub(" ", text)
    text = UNINTELL_PAREN.sub(" ", text)
    if drop_chat_extra:
        text = _PAREN_GROUP.sub(" ", text)
        text = _CHAT_EXTRA.sub(" ", text)
    text = text.replace(" ... ", " ")
    text = text.replace("#x27;", "'")
    text = REPL_PUNC.sub(" ", text)
    # Remove sentence periods but preserve "..." ellipses (parked behind a sentinel).
    text = (
        text.replace("...", "!ELLIPSIS!")
        .replace(".", " ")
        .replace("!ELLIPSIS!", "...")
    )
    text = MULTISPACE.sub(" ", text)
    return text.strip()


def _score_clean(text: str) -> str:
    """clean_transcript + lowercase — what the WER scorer compares."""
    return clean_transcript(text).lower()


def normalize_es_en(text: str) -> str:
    return _score_clean(text)


def normalize_id_jv(text: str) -> str:
    return _score_clean(text)


def normalize_nahuatl(text: str) -> str:
    return to_uks_orthography(_score_clean(text))


_PROFILES = {
    "es_en": normalize_es_en,
    "id_jv": normalize_id_jv,
    "nahuatl": normalize_nahuatl,
    "default": normalize_es_en,
}


def normalize(text: str, profile: str = "default") -> str:
    """Normalize `text` with the named profile (see configs/tracks.yaml `normalize`)."""
    if text is None:
        return ""
    return _PROFILES.get(profile, _PROFILES["default"])(text)
