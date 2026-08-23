"""Config loading for tracks (configs/tracks.yaml)."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "tracks.yaml"


def load_tracks(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    defaults = raw.get("defaults", {})
    tracks = {}
    for name, tcfg in raw["tracks"].items():
        merged = copy.deepcopy(defaults)
        # deep-merge the nested `split` block, shallow-merge the rest
        split = {**merged.get("split", {}), **tcfg.get("split", {})}
        merged.update(tcfg)
        merged["split"] = split
        merged["track"] = name
        tracks[name] = merged
    return tracks


def get_track(name: str, path: str | Path = DEFAULT_CONFIG) -> dict:
    tracks = load_tracks(path)
    if name not in tracks:
        raise KeyError(f"unknown track '{name}'; known: {list(tracks)}")
    return tracks[name]
