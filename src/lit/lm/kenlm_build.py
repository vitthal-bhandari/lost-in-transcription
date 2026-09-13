"""Build an n-gram KenLM from normalized training text (for CTC beam-search decoding).

Shells out to the `lmplz`/`build_binary` CLI tools (NOT the pip `kenlm` package, which only
ships Python bindings for decoding — see scripts/slurm/setup_kenlm_tools.sh). Trains on text
already passed through the SAME normalization the decoder targets (official normalize_text +
fold_diacritics), so the LM biases decoding toward the exact dev/test convention.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BIN_DIR = REPO_ROOT / "tools" / "kenlm" / "bin"


def _find_tool(name: str) -> str:
    local = DEFAULT_BIN_DIR / name
    if local.exists():
        return str(local)
    on_path = shutil.which(name)
    if on_path:
        return on_path
    raise FileNotFoundError(
        f"{name} not found at {local} or on PATH. Run scripts/slurm/setup_kenlm_tools.sh first."
    )


def build_lm(train_texts: list[str], out_dir: str | Path, order: int = 4) -> Path:
    """Train an order-N KenLM from `train_texts` (one utterance per element, already normalized).

    Writes lm_train.txt, lm_{order}gram.arpa, lm_{order}gram.bin under out_dir. Returns the
    path to the binary LM (what pyctcdecode consumes at decode time — faster + smaller than
    the .arpa).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_txt = out / "lm_train.txt"
    with open(train_txt, "w") as f:
        for line in train_texts:
            line = line.strip()
            if line:
                f.write(line + "\n")
    print(f"[kenlm] wrote {sum(1 for _ in open(train_txt))} lines -> {train_txt}")

    lmplz, build_binary = _find_tool("lmplz"), _find_tool("build_binary")
    arpa_path = out / f"lm_{order}gram.arpa"
    bin_path = out / f"lm_{order}gram.bin"

    with open(train_txt) as fin, open(arpa_path, "w") as fout:
        subprocess.run(
            [lmplz, "-o", str(order), "--discount_fallback"],
            stdin=fin, stdout=fout, check=True,
        )
    print(f"[kenlm] trained {order}-gram ARPA -> {arpa_path}")

    subprocess.run([build_binary, str(arpa_path), str(bin_path)], check=True)
    print(f"[kenlm] built binary LM -> {bin_path}")
    return bin_path
