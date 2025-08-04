#!/usr/bin/env bash
# -------------------------------------------------------------
# Save any ESM‑2 checkpoint locally with the modern HF CLI.
#
# USAGE
#   ./hf_download_esm2.sh [MODEL_ID] [DEST_DIR]
# EXAMPLE
#   ./hf_download_esm2.sh facebook/esm2_t33_650M_UR50D ./models
# -------------------------------------------------------------
set -euo pipefail

MODEL_ID="${1:-togethercomputer/evo-1-131k-base}"
DEST_DIR="${2:-./models}"
TARGET_DIR="${DEST_DIR}/$(basename "$MODEL_ID")"

mkdir -p "$DEST_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1   # concurrent & resumable

if [[ -d "$TARGET_DIR" ]]; then
  echo "✔ $TARGET_DIR already exists – skipped."
  exit 0
fi

echo "⬇  Downloading ${MODEL_ID} …"
hf download "${MODEL_ID}" --local-dir "${TARGET_DIR}"

echo "✅ Model stored at ${TARGET_DIR}"
