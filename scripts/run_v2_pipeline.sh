#!/usr/bin/env bash
# Build v2 composite targets, cache their latents, and train v2.
# Inputs are prepared outside this repository: --proc must contain
# manifest.jsonl plus src/, tgt/, and mask/, while --ocr contains OCR JSON.
# MTDIFF_VAE is required because the FLUX VAE is not bundled.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${MTDIFF_DATA_ROOT:-$ROOT/data}"
PROC="${MTDIFF_PROC:-$DATA_ROOT/proc}"
OCR="${MTDIFF_OCR:-$DATA_ROOT/ocr}"
REUSE_CACHE="${MTDIFF_REUSE_CACHE_DIR:-$DATA_ROOT/latents}"
CACHE="${MTDIFF_CACHE_DIR:-$DATA_ROOT/latents_v2}"
RUNS="${MTDIFF_RUNS_DIR:-$ROOT/runs}"
VAE="${MTDIFF_VAE:-}"
STEPS="${1:-60000}"

if [[ -z "$VAE" ]]; then echo "MTDIFF_VAE must point to the local FLUX VAE directory" >&2; exit 2; fi
if [[ ! -f "$PROC/manifest.jsonl" ]]; then echo "missing prepared manifest: $PROC/manifest.jsonl" >&2; exit 2; fi
if [[ ! -d "$OCR" ]]; then echo "missing OCR directory: $OCR" >&2; exit 2; fi
if [[ ! -d "$REUSE_CACHE" ]]; then echo "missing v1 latent cache for --reuse-src: $REUSE_CACHE" >&2; exit 2; fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
python scripts/build_v2.py --proc "$PROC" --ocr "$OCR"
python scripts/cache_latents.py --proc "$PROC" --manifest manifest_v2.jsonl --out "$CACHE" --reuse-src "$REUSE_CACHE" --vae "$VAE"
python train.py --run-dir "$RUNS/v2" --manifest "$PROC/manifest_v2.jsonl" --cache-dir "$CACHE" --steps "$STEPS" --synthetic-fraction 0 --vae-path "$VAE"
