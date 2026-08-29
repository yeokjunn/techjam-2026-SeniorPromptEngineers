#!/bin/sh
# Fetch the KuaiRand-Pure dataset into data/.
#
# The dataset is not tracked in git (see .gitignore). Run this once after cloning,
# or after a pull that removed data/ from your working tree.
#
# Usage: sh scripts/download_data.sh
set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DATA_DIR="$REPO_ROOT/data"
SENTINEL="$DATA_DIR/KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv"
ARCHIVE="$DATA_DIR/KuaiRand-Pure.tar.gz"
URL="https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
EXPECTED_MD5="0820331067a3784d9691136f772b35a7"

if [ -f "$SENTINEL" ]; then
    echo "already present: data/KuaiRand-Pure/data"
    exit 0
fi

mkdir -p "$DATA_DIR"

echo "downloading KuaiRand-Pure (~280 MB) ..."
if command -v curl >/dev/null 2>&1; then
    curl -fL -o "$ARCHIVE" "$URL"
elif command -v wget >/dev/null 2>&1; then
    wget -O "$ARCHIVE" "$URL"
else
    echo "need curl or wget on PATH" >&2
    exit 1
fi

echo "verifying checksum ..."
if command -v md5sum >/dev/null 2>&1; then
    actual_md5=$(md5sum "$ARCHIVE" | cut -d ' ' -f 1)
elif command -v md5 >/dev/null 2>&1; then
    actual_md5=$(md5 -q "$ARCHIVE")
else
    echo "need md5sum (Linux) or md5 (macOS) on PATH" >&2
    exit 1
fi

if [ "$actual_md5" != "$EXPECTED_MD5" ]; then
    # Keep the archive so a partial or tampered download can be inspected.
    echo "checksum mismatch: expected $EXPECTED_MD5, got $actual_md5" >&2
    echo "leaving $ARCHIVE in place; delete it and re-run to retry" >&2
    exit 1
fi

echo "extracting ..."
tar -xzf "$ARCHIVE" -C "$DATA_DIR"
rm -f "$ARCHIVE"

echo "done: data/KuaiRand-Pure/data"
