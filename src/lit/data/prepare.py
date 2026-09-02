"""Per-corpus preparation into a unified manifest.

Each corpus ships in a different format. Each `prepare_<track>` returns a DataFrame with the
unified schema so everything downstream (splits, training, scoring) is corpus-agnostic.

Unified manifest schema:
    audio_path : str    path to the audio file (a clip, or a long recording to be sliced)
    text       : str    reference transcript (raw; normalization happens at scoring/train time)
    speaker    : str    speaker id (for disjoint splits); may be NaN if unknown
    session    : str    conversation/recording id (grouping key for disjoint splits)
    duration   : float  seconds
    start      : float  segment start in seconds within audio_path (NaN if audio_path is already
                        a clip); when present, the loader slices [start, end)
    end        : float  segment end in seconds within audio_path (NaN for clips)
    language   : str    corpus language tag if provided (e.g. "javind"), else NaN
    subset     : str    provenance: "train_corpus" | "dev" (official) | "test"
    split      : str    train | val | dev | test  (corpus-defined; see notes per track)
    track      : str    es_en | es_nah | id_jv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MANIFEST_COLUMNS = [
    "audio_path", "text", "speaker", "session", "duration",
    "start", "end", "language", "subset", "split", "track",
]


def _empty_manifest() -> pd.DataFrame:
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def _hms_to_seconds(value: str) -> float:
    """Parse 'H:MM:SS' / 'H:MM:SS.mmm' / 'MM:SS' timestamps to float seconds."""
    parts = str(value).strip().split(":")
    parts = [float(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0.0)  # pad missing hours/minutes
    h, m, s = parts[-3], parts[-2], parts[-1]
    return h * 3600 + m * 60 + s


# --------------------------------------------------------------------------- es_en / es_nah stubs
def prepare_es_en(data_dir: str | Path) -> pd.DataFrame:
    raise NotImplementedError("prepare_es_en: implement once Bangor Miami layout is known")


def prepare_es_nah(data_dir: str | Path) -> pd.DataFrame:
    raise NotImplementedError("prepare_es_nah: implement once Nahuatl corpus layout is known")


# --------------------------------------------------------------------------- id_jv (Jember)
def _find_one(root: Path, pattern: str) -> Path:
    hits = sorted(root.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"no file matching {pattern!r} under {root}")
    return hits[0]


def _prepare_jember_train(root: Path) -> pd.DataFrame:
    """200 long-form recordings + a TSV of utterance segments (Audio file name, start, end, text).

    audio_path points at the whole recording; start/end mark the utterance to slice at load time.
    `session` = recording id (no speaker column in this TSV, so recording is our grouping key).
    """
    tsv = _find_one(root, "*1-200.tsv")
    audio_dir = tsv.parent / "mp3 audio"
    df = pd.read_csv(tsv, sep="\t", dtype={"Audio file name": str})
    df = df.rename(columns={"Audio file name": "rec", "text": "text"})
    df = df[df["text"].notna() & (df["text"].astype(str).str.strip() != "")].copy()

    df["start"] = df["start"].map(_hms_to_seconds)
    df["end"] = df["end"].map(_hms_to_seconds)
    # Timestamps are integer-second granularity, so utterances that start and end within the same
    # second collapse to 0s. Extend such degenerate windows to 1s to recover the (real) audio
    # rather than dropping ~41 valid short utterances ("Polusi.", "Ha'a.", ...).
    degenerate = df["end"] <= df["start"]
    if degenerate.any():
        print(f"[id_jv/train] padding {int(degenerate.sum())} zero-length segments to 1s windows")
        df.loc[degenerate, "end"] = df.loc[degenerate, "start"] + 1.0
    df["duration"] = (df["end"] - df["start"]).clip(lower=0.0)
    df["audio_path"] = df["rec"].map(lambda r: str(audio_dir / f"{r}.mp3"))
    df["session"] = "rec_" + df["rec"].astype(str)
    df["speaker"] = pd.NA          # not provided at utterance level in the train TSV
    df["language"] = "javind"
    df["subset"] = "train_corpus"
    df["split"] = pd.NA            # filled by holdout split below
    df["track"] = "id_jv"

    # Drop segments whose recording file is missing on disk.
    present = df["audio_path"].map(lambda p: Path(p).exists())
    missing = int((~present).sum())
    if missing:
        print(f"[id_jv/train] dropping {missing} segments with missing recording files")
    return df[present][MANIFEST_COLUMNS].reset_index(drop=True)


def _prepare_indonesian_dev(root: Path) -> pd.DataFrame:
    """372 pre-segmented clips + metadata.tsv (audio_filename, speaker, transcript, language,
    convo_id). Same format as the hidden test set -> our honest eval proxy."""
    tsv = _find_one(root, "indonesian_dev/metadata.tsv")
    clips_dir = tsv.parent / "clips"
    df = pd.read_csv(tsv, sep="\t", dtype=str)
    df = df.rename(columns={"transcript": "text"})
    df["audio_path"] = df["audio_filename"].map(lambda f: str(clips_dir / f))
    df["session"] = "convo_" + df["convo_id"].astype(str)
    df["speaker"] = "spk_" + df["speaker"].astype(str)
    df["start"] = pd.NA           # already clipped
    df["end"] = pd.NA
    df["duration"] = pd.NA        # probe later if needed (not required: dev is not re-split)
    df["subset"] = "dev"
    df["split"] = "dev"
    df["track"] = "id_jv"
    return df[MANIFEST_COLUMNS].reset_index(drop=True)


def prepare_id_jv(data_dir: str | Path) -> pd.DataFrame:
    """Build the Jember Javanese-Indonesian manifest: sliced train segments + official dev clips.

    The official `dev` set matches the hidden-test distribution, so we keep it whole as our eval
    proxy and split the training corpus into train/val by *recording* (session-disjoint) later in
    create_splits.py. Returns rows with subset in {train_corpus, dev}; train rows have split=NA.
    """
    root = Path(data_dir)
    train_df = _prepare_jember_train(root)
    dev_df = _prepare_indonesian_dev(root)
    return pd.concat([train_df, dev_df], ignore_index=True)


PREPARERS = {
    "es_en": prepare_es_en,
    "es_nah": prepare_es_nah,
    "id_jv": prepare_id_jv,
}
