#!/bin/bash
# ============================================================
# postprocess_loo.sh — OCR post-processing for LOO inference results
#
# Usage:
# bash scripts/postprocess_loo.sh <loo_dir> # a single LOO dir
# bash scripts/postprocess_loo.sh --all # process all LOO dirs
# bash scripts/postprocess_loo.sh <loo_dir> --max-dist 3 --skip-condition
#
# By default it: (1) OCR-corrects the triples, (2) re-scores via lora/eval/eval_TRACE.py,
# (3) prints a before/after comparison. Add --no-eval to only correct.
#
# Dependencies:
# pip install paddleocr python-Levenshtein opencv-python pillow
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# Pick python: prefer the sam3 env (has paddleocr + Levenshtein), else fall back to current python
SAM3_PY="<CONDA_ROOT>/envs/<CONDA_ENV>/bin/python"
if [ -x "$SAM3_PY" ] && "$SAM3_PY" -c "import paddleocr, Levenshtein" 2>/dev/null; then
    PY="$SAM3_PY"
else
    PY="python"
fi
echo "[run_loo] python: $PY"

if [ $# -eq 0 ]; then
    echo "Usage:"
    echo "  bash scripts/postprocess_loo.sh <loo_dir> [--max-dist N] [--skip-condition] [--no-eval]"
    echo "  bash scripts/postprocess_loo.sh --all     [--max-dist N] [--skip-condition] [--no-eval]"
    exit 1
fi

# First arg: a single dir path, or --all
ARGS=()
if [ "$1" = "--all" ]; then
    ARGS+=("--all")
else
    ARGS+=("--loo-dir" "$1")
fi
shift
ARGS+=("$@")

"$PY" -m postprocess.correct_loo "${ARGS[@]}"
