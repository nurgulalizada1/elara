#!/usr/bin/env bash
# Quick end-to-end smoke test against an isolated data dir. Usage: scripts/smoke.sh
set -euo pipefail
TMP=$(mktemp -d)
export ELARA_DATA_DIR="$TMP/data" ELARA_WRITE_DIRS="[\"$TMP/ws\"]" ELARA_READ_DIRS="[\"$TMP/ws\"]"
mkdir -p "$TMP/ws"
elara ask "What's 17 * 42?"
elara ask "Remember that my favorite programming language is Python."
elara ask "What's my favorite programming language?"
elara memory list
elara doctor --offline || true
rm -rf "$TMP"
