"""Bridge to the OFFICIAL competition scorer (runtime/score.py, a git submodule).

We import `normalize_text` directly from the official script so local WER uses the exact same
normalization as the leaderboard — no reimplementation, no drift. If the official normalization
changes upstream, `git submodule update --remote runtime` picks it up.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCORE_PY = REPO_ROOT / "runtime" / "score.py"


@lru_cache(maxsize=1)
def _official_module():
    if not SCORE_PY.exists():
        raise FileNotFoundError(
            f"official scorer not found at {SCORE_PY}. Initialize the submodule:\n"
            "  git submodule update --init runtime"
        )
    spec = importlib.util.spec_from_file_location("lit_official_score", SCORE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # top-level imports: re, jiwer, pandas, typer (all in our deps)
    return mod


def normalize_text(text: str) -> str:
    """The competition's exact normalization (from runtime/score.py)."""
    return _official_module().normalize_text("" if text is None else str(text))


def wer(refs: list[str], hyps: list[str]) -> float:
    """Corpus WER via the official normalization + jiwer.wer — identical math to score.py."""
    import jiwer

    norm_refs = [normalize_text(r) for r in refs]
    norm_hyps = [normalize_text(h) for h in hyps]
    return jiwer.wer(norm_refs, norm_hyps)
