#!/usr/bin/env bash
# Build KenLM's CLI tools (lmplz, build_binary) from source. The `kenlm` PyPI package only
# ships the Python bindings (used for DECODING via pyctcdecode) — it does NOT install lmplz/
# build_binary, which we need to BUILD an .arpa/.bin LM from our own training text. One-time,
# CPU-only (fine on a login node); needs cmake + a C++ compiler (both present on Tillicum, same
# toolchain that already built the pip kenlm wheel during the omni env setup).
#
# Run from repo root: bash scripts/slurm/setup_kenlm_tools.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO_DIR="$(pwd)"

CACHE_ROOT="${LIT_CACHE_ROOT:-/gpfs/scrubbed/$USER/.cache}"
SRC_DIR="$CACHE_ROOT/kenlm_src"
BIN_DIR="$REPO_DIR/tools/kenlm/bin"
mkdir -p "$SRC_DIR" "$BIN_DIR"

if [ ! -x "$BIN_DIR/lmplz" ]; then
  echo ">>> cloning + building kenlm CLI tools into $SRC_DIR ..."
  [ -d "$SRC_DIR/kenlm" ] || git clone --depth 1 https://github.com/kpu/kenlm.git "$SRC_DIR/kenlm"
  mkdir -p "$SRC_DIR/kenlm/build"
  cd "$SRC_DIR/kenlm/build"
  cmake .. -DCMAKE_BUILD_TYPE=Release
  make -j"$(nproc)" lmplz build_binary
  cp bin/lmplz bin/build_binary "$BIN_DIR/"
  cd "$REPO_DIR"
else
  echo ">>> lmplz already built at $BIN_DIR/lmplz"
fi

"$BIN_DIR/lmplz" --help >/dev/null 2>&1 && echo ">>> lmplz OK"
"$BIN_DIR/build_binary" 2>&1 | head -1 || true
echo ">>> kenlm tools ready: $BIN_DIR (add to PATH, or lit.lm.kenlm_build finds it automatically)"
