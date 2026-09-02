#!/usr/bin/env bash
# Pack submission_src/ into submission/submission.zip using rpzip (reproducible zip), matching the
# official runtime's packing flow (runtime/examples/template/pack_submission.sh).
#
# Usage: bash submission_src/pack_submission.sh [output_dir]
# The zip must contain main.py at its ROOT plus any model weights under assets/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$REPO_ROOT/submission}"
mkdir -p "$OUTPUT_DIR"

cd "$REPO_ROOT/submission_src"
echo "Packing $(pwd)/* -> $OUTPUT_DIR/submission.zip"
uvx rpzip -r "$OUTPUT_DIR/submission.zip" ./*
echo "done:"
ls -lh "$OUTPUT_DIR/submission.zip"
