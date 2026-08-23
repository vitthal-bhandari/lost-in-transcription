"""Per-corpus preparation into a unified manifest.

Each corpus ships in a different format (Bangor Miami = CHAT/CLAN .cha transcripts; Nahuatl =
ELAN/text; Jember = its own layout). Each `prepare_<track>` returns a DataFrame with the unified
schema so everything downstream (splits, training, scoring) is corpus-agnostic.

Unified manifest schema:
    audio_path : str   absolute or data-dir-relative path to the audio file
    text       : str   raw reference transcript (normalization happens at scoring/training time)
    speaker    : str   speaker id (for disjoint splits)
    session    : str   conversation/recording id
    duration   : float seconds (for duration-weighted splits + length filtering)
    track      : str   es_en | es_nah | id_jv

These are STUBS pending the real data layout (login-gated). Fill in once data is downloaded.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MANIFEST_COLUMNS = ["audio_path", "text", "speaker", "session", "duration", "track"]


def _empty_manifest() -> pd.DataFrame:
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def prepare_es_en(data_dir: str | Path) -> pd.DataFrame:
    """Bangor Miami: parse CHAT/CLAN .cha files → utterance manifest.

    TODO: parse .cha tiers; segment by utterance timestamps; map @ID speaker codes.
    """
    raise NotImplementedError("prepare_es_en: implement once Bangor Miami layout is known")


def prepare_es_nah(data_dir: str | Path) -> pd.DataFrame:
    """W. Sierra Puebla Nahuatl: 2,681 pre-segmented utterances → manifest.

    TODO: map audio↔transcript; recover speaker ids for grouped k-fold.
    """
    raise NotImplementedError("prepare_es_nah: implement once Nahuatl corpus layout is known")


def prepare_id_jv(data_dir: str | Path) -> pd.DataFrame:
    """Jember Javanese: 6,679 utterances → manifest.

    TODO: map audio↔transcript + speaker/session; note 4-way code-mixing in `text`.
    """
    raise NotImplementedError("prepare_id_jv: implement once Jember layout is known")


PREPARERS = {
    "es_en": prepare_es_en,
    "es_nah": prepare_es_nah,
    "id_jv": prepare_id_jv,
}
