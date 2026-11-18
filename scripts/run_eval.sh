#!/usr/bin/env bash
# Evaluate a training checkpoint with OCR read-back.
# Usage: run_eval.sh CHECKPOINT TAG [pred|target|source ...]
# predict_pages.py reads manifest/cache/VAE paths embedded in the checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_ROOT="${MTDIFF_EVAL_DIR:-$ROOT/eval}"
if [[ $# -lt 2 ]]; then echo "usage: $0 CHECKPOINT TAG [pred|target|source ...]" >&2; exit 2; fi
CKPT="$1"
TAG="$2"
shift 2
if [[ ! -f "$CKPT" ]]; then echo "checkpoint not found: $CKPT" >&2; exit 2; fi
if (( $# == 0 )); then set -- pred; fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$EVAL_ROOT"
cd "$ROOT"
for MODE in "$@"; do
    case "$MODE" in
        pred|target|source) ;;
        *) echo "invalid mode '$MODE'; expected pred, target, or source" >&2; exit 2 ;;
    esac
    NAME="${TAG}_${MODE}"
    IMAGE_DIR="$EVAL_ROOT/$NAME"
    python scripts/predict_pages.py --checkpoint "$CKPT" --out "$IMAGE_DIR" --mode "$MODE"
    python scripts/ocr_eval.py --images "$IMAGE_DIR" --proc "${MTDIFF_PROC:-$ROOT/data/proc}" --ocr "${MTDIFF_OCR:-$ROOT/data/ocr}" --out "$EVAL_ROOT/$NAME.json"
done
echo "evaluation finished: $TAG"
