#!/usr/bin/env bash
# Download a Mozilla Data Collective dataset via its presigned-URL API.
#
# Usage:
#   scripts/download_data.sh <dataset_id> <output_filename.tar.gz>
#
# The API key is read from $MDC_API_KEY (export it, or put it in a gitignored .env at repo root
# as: MDC_API_KEY=...). The key is never printed. Tarballs land in data/raw/.
set -euo pipefail

DATASET_ID="${1:?usage: download_data.sh <dataset_id> <output_filename.tar.gz>}"
OUT_NAME="${2:?usage: download_data.sh <dataset_id> <output_filename.tar.gz>}"

# Load .env if present (without echoing it).
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a; source "$REPO_ROOT/.env"; set +a
fi
: "${MDC_API_KEY:?set MDC_API_KEY (in env or repo-root .env)}"

RAW_DIR="$REPO_ROOT/data/raw"
mkdir -p "$RAW_DIR"

echo "[1/3] Requesting presigned URL for dataset $DATASET_ID ..."
RESPONSE="$(curl -sS -X POST \
  "https://mozilladatacollective.com/api/datasets/${DATASET_ID}/download" \
  -H "Authorization: Bearer ${MDC_API_KEY}" \
  -H "Content-Type: application/json")"

DOWNLOAD_URL="$(echo "$RESPONSE" | jq -r '.downloadUrl // empty')"
if [[ -z "$DOWNLOAD_URL" ]]; then
  echo "ERROR: no downloadUrl in API response:" >&2
  echo "$RESPONSE" | jq . >&2 2>/dev/null || echo "$RESPONSE" >&2
  exit 1
fi

# Report size before downloading (presigned URLs expose Content-Length via a HEAD).
SIZE="$(curl -sSI "$DOWNLOAD_URL" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r')"
if [[ -n "${SIZE:-}" ]]; then
  echo "[2/3] Remote size: $(numfmt --to=iec "$SIZE" 2>/dev/null || echo "$SIZE bytes")"
fi

echo "[3/3] Downloading -> data/raw/$OUT_NAME"
curl -# -o "$RAW_DIR/$OUT_NAME" "$DOWNLOAD_URL"
echo "done: $RAW_DIR/$OUT_NAME"
ls -lh "$RAW_DIR/$OUT_NAME"
