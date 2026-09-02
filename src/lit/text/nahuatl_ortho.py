"""Nahuatl orthography normalization to the competition's (u, k, s) conventions.

The es_nah track requires a specific orthography: /k/ written as `k` (not `c`/`qu`),
/s/ written as `s` (not `c`/`z`), and `u` where classical spelling uses `o`/`hu`/`uh`.
This is a FIRST-PASS mapping — verify against the official Nahuatl orthography guide and
the corpus's own conventions once we have the data + scorer, then refine here.
"""

from __future__ import annotations

import re

# Order matters: apply multi-char rules before single-char.
_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"qu(?=[ei])"), "k"),   # que/qui -> ke/ki
    (re.compile(r"c(?=[ei])"), "s"),    # ce/ci   -> se/si  (/s/ before front vowels)
    (re.compile(r"c(?=[aou])"), "k"),   # ca/co/cu-> ka/ko/ku
    (re.compile(r"c$"), "k"),           # syllable/word-final c -> k
    (re.compile(r"z"), "s"),            # z -> s
    (re.compile(r"hu"), "u"),           # hu -> u  (revisit: /w/ handling)
    (re.compile(r"uh"), "u"),
]


def to_uks_orthography(text: str) -> str:
    for pattern, repl in _RULES:
        text = pattern.sub(repl, text)
    return text
