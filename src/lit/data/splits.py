"""Speaker/session-disjoint and k-fold splitting.

Conversational corpora leak badly under random utterance splits: the same speaker (and often
the same conversation) lands in train and eval, inflating WER optimism. We ALWAYS hold out
whole speakers or sessions. For the tiny es_nah track we use speaker-grouped k-fold so a
single unlucky split doesn't mislead us.

A manifest is a DataFrame with at least: audio_path, text, speaker, session, duration.
"""

from __future__ import annotations

import hashlib

import pandas as pd


def _stable_group_order(groups: list[str], seed: int) -> list[str]:
    """Deterministic shuffle of group ids seeded by `seed` (hash-based, reproducible)."""
    def key(g: str) -> str:
        return hashlib.md5(f"{seed}:{g}".encode()).hexdigest()
    return sorted(groups, key=key)


def speaker_disjoint_split(
    manifest: pd.DataFrame,
    dev_frac: float = 0.10,
    test_frac: float = 0.10,
    group_col: str = "speaker",
    seed: int = 13,
) -> pd.DataFrame:
    """Add a `split` column (train/dev/test) with disjoint `group_col` values.

    Allocation is by cumulative *duration* (falls back to utterance count if no duration),
    so dev/test get ~the requested fraction of audio rather than of speakers.
    """
    df = manifest.copy()
    weight = df["duration"] if "duration" in df.columns else pd.Series(1, index=df.index)
    by_group = weight.groupby(df[group_col]).sum()
    total = by_group.sum()

    ordered = _stable_group_order(list(by_group.index.astype(str)), seed)
    test_target, dev_target = test_frac * total, dev_frac * total

    assign, cum_test, cum_dev = {}, 0.0, 0.0
    for g in ordered:
        w = float(by_group.loc[g])
        if cum_test < test_target:
            assign[g], cum_test = "test", cum_test + w
        elif cum_dev < dev_target:
            assign[g], cum_dev = "dev", cum_dev + w
        else:
            assign[g] = "train"
    df["split"] = df[group_col].astype(str).map(assign)
    return df


def speaker_kfold(
    manifest: pd.DataFrame,
    k: int = 5,
    group_col: str = "speaker",
    seed: int = 13,
) -> pd.DataFrame:
    """Add a `fold` column (0..k-1) with disjoint `group_col` values per fold."""
    df = manifest.copy()
    ordered = _stable_group_order(list(df[group_col].astype(str).unique()), seed)
    fold_of = {g: i % k for i, g in enumerate(ordered)}
    df["fold"] = df[group_col].astype(str).map(fold_of)
    return df


def split_train_corpus(
    manifest: pd.DataFrame,
    val_frac: float = 0.10,
    group_col: str = "session",
    seed: int = 13,
) -> pd.DataFrame:
    """Assign train/val (group-disjoint) to rows whose `split` is NA; leave assigned rows intact.

    Used when a corpus ships its own held-out set (e.g. the official Jember dev): those rows keep
    their split (e.g. "dev") and only the training pool is carved into train/val here, disjoint by
    `group_col` (recording/session) and duration-weighted so val gets ~val_frac of the audio.
    """
    df = manifest.copy()
    if "split" not in df.columns:
        df["split"] = pd.NA
    mask = df["split"].isna()
    if not mask.any():
        return df

    sub = df[mask]
    weight = sub["duration"].fillna(1.0) if "duration" in sub.columns else pd.Series(1.0, index=sub.index)
    by_group = weight.groupby(sub[group_col]).sum()
    total = by_group.sum()
    ordered = _stable_group_order(list(by_group.index.astype(str)), seed)

    val_target, cum = val_frac * total, 0.0
    assign = {}
    for g in ordered:
        if cum < val_target:
            assign[g], cum = "val", cum + float(by_group.loc[g])
        else:
            assign[g] = "train"
    df.loc[mask, "split"] = sub[group_col].astype(str).map(assign).values
    return df


def make_splits(manifest: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # Corpus-defined splits present (some rows already labeled): only carve train/val from the
    # unassigned training pool, preserving the shipped held-out set.
    if "split" in manifest.columns and manifest["split"].notna().any():
        return split_train_corpus(
            manifest, val_frac=cfg.get("dev_frac", 0.10), group_col="session", seed=cfg.get("seed", 13)
        )

    """Dispatch on cfg['strategy'] (speaker_disjoint | session_disjoint | kfold)."""
    strategy = cfg.get("strategy", "speaker_disjoint")
    seed = cfg.get("seed", 13)
    if strategy == "kfold":
        return speaker_kfold(manifest, k=cfg.get("kfold", 5), group_col="speaker", seed=seed)
    group_col = "session" if strategy == "session_disjoint" else "speaker"
    return speaker_disjoint_split(
        manifest,
        dev_frac=cfg.get("dev_frac", 0.10),
        test_frac=cfg.get("test_frac", 0.10),
        group_col=group_col,
        seed=seed,
    )
