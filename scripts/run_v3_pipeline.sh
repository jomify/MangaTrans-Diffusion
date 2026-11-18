#!/usr/bin/env bash
# Build synthetic v3 variants, cache their latents, and train from v2.
# Requires v2 outputs from run_v2_pipeline.sh and prepared OCR/processed data.
# MTDIFF_VAE, MTDIFF_DATA_ROOT, MTDIFF_PROC, MTDIFF_OCR,
# MTDIFF_CACHE_DIR, MTDIFF_REUSE_CACHE_DIR, and MTDIFF_RUNS_DIR override
# the portable defaults.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${MTDIFF_DATA_ROOT:-$ROOT/data}"
PROC="${MTDIFF_PROC:-$DATA_ROOT/proc}"
OCR="${MTDIFF_OCR:-$DATA_ROOT/ocr}"
REUSE_CACHE="${MTDIFF_REUSE_CACHE_DIR:-$DATA_ROOT/latents}"
CACHE="${MTDIFF_CACHE_DIR:-$DATA_ROOT/latents_v2}"
RUNS="${MTDIFF_RUNS_DIR:-$ROOT/runs}"
VAE="${MTDIFF_VAE:-}"
STEPS="${1:-40000}"
VARIANTS="${2:-4}"

if [[ -z "$VAE" ]]; then echo "MTDIFF_VAE must point to the local FLUX VAE directory" >&2; exit 2; fi
if [[ ! -f "$PROC/manifest_v2.jsonl" ]]; then echo "missing v2 manifest: $PROC/manifest_v2.jsonl (run run_v2_pipeline.sh first)" >&2; exit 2; fi
if [[ ! -d "$OCR" ]]; then echo "missing OCR directory: $OCR" >&2; exit 2; fi
if [[ ! -d "$REUSE_CACHE" ]]; then echo "missing v1 latent cache for --reuse-src: $REUSE_CACHE" >&2; exit 2; fi
if [[ ! -f "$RUNS/v2/latest.pt" ]]; then echo "missing v2 checkpoint: $RUNS/v2/latest.pt (run run_v2_pipeline.sh first)" >&2; exit 2; fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
python scripts/build_synth.py --proc "$PROC" --ocr "$OCR" --variants "$VARIANTS"
python scripts/cache_latents.py --proc "$PROC/syn" --manifest manifest_syn.jsonl --out "$CACHE" --reuse-src "$REUSE_CACHE" --vae "$VAE"
cat "$PROC/manifest_v2.jsonl" "$PROC/syn/manifest_syn.jsonl" > "$PROC/manifest_v3.jsonl"
python train.py --run-dir "$RUNS/v3" --manifest "$PROC/manifest_v3.jsonl" --cache-dir "$CACHE" --steps "$STEPS" --init-from "$RUNS/v2/latest.pt" --warmup-steps 1000 --vae-path "$VAE"
