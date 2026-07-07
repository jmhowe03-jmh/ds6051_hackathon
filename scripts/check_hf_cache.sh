#!/usr/bin/env bash
# Show what's cached locally so you know whether a model needs downloading
# before you burn GPU time / login-node bandwidth on it.
# cd into hackathon folder to run this script, or set HF_HOME to the cache directory you want to check.
set -euo pipefail

CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

echo "Cache location: $CACHE_DIR"
echo "Total size:"
du -sh "$CACHE_DIR" 2>/dev/null || echo "  (nothing cached yet)"
echo

echo "Cached repos:"
uv run hf cache list || echo "  (no cache directory found — nothing downloaded yet)"
