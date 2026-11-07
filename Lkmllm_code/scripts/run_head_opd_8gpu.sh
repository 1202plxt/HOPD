#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CODE_DIR="$PWD"
ASSET_ROOT="${LK_OPD_ROOT:-}"

# A git clone adds HOPD/ between the code and the user's existing assets.
# Find their common parent instead of assuming they are inside the clone.
if [[ -z "$ASSET_ROOT" ]]; then
  candidate="$CODE_DIR"
  while [[ "$candidate" != / ]]; do
    if [[ -d "$candidate/shared_models" && -d "$candidate/Lkmllm_data" ]]; then
      ASSET_ROOT="$candidate"
      break
    fi
    candidate="$(dirname "$candidate")"
  done
fi
if [[ -n "$ASSET_ROOT" ]]; then
  ASSET_ROOT="$(cd "$ASSET_ROOT" && pwd)"
elif [[ -z "${STUDENT:-}" || -z "${TEACHER:-}" || -z "${ANNO:-}" || -z "${VIDEOS:-}" || -z "${OUT:-}" ]]; then
  echo 'Cannot locate shared_models/ and Lkmllm_data/. Set LK_OPD_ROOT, or all of STUDENT/TEACHER/ANNO/VIDEOS/OUT.' >&2
  exit 2
fi

STUDENT="${STUDENT:-$ASSET_ROOT/shared_models/Qwen3-VL-8B-Instruct}"
TEACHER="${TEACHER:-$ASSET_ROOT/shared_models/Video-opd-teacher-Qwen3-VL-32B-GRPO}"
ANNO="${ANNO:-$ASSET_ROOT/Lkmllm_data/datasets/Train/timelens-100k/timelens-100k.jsonl}"
VIDEOS="${VIDEOS:-$ASSET_ROOT/Lkmllm_data/datasets/Train/timelens-100k}"
OUT="${OUT:-$ASSET_ROOT/Lkmllm_data/checkpoints/qwen3_top10_independent_seconds_opd_2k_$(date +%Y%m%d_%H%M%S)}"
PY="${PY:-python}"
command -v "$PY" >/dev/null || { echo "Python not found: $PY" >&2; exit 2; }

# Detect the backend before the trainer initializes its visible GPU topology.
BACKEND="$("$PY" -c 'import torch; print("hip" if torch.version.hip else "cuda" if torch.version.cuda else "none")')"
case "$BACKEND" in
  hip) unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES HSA_VISIBLE_DEVICES ;;
  cuda) unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES HSA_VISIBLE_DEVICES ;;
  *) echo 'Selected Python has no CUDA/HIP PyTorch. Set PY to the training environment interpreter.' >&2; exit 2 ;;
esac

extra=()
case "${CPU_OFFLOAD:-1}" in
  1) extra+=(--offload-saved-tensors) ;;
  0) ;;
  *) echo 'CPU_OFFLOAD must be 0 or 1' >&2; exit 2 ;;
esac
echo "Python: $PY; backend: $BACKEND; CPU offload: ${CPU_OFFLOAD:-1}"
echo "Asset root: ${ASSET_ROOT:-explicit paths}"
echo "Training output: $OUT"
exec "$PY" -u scripts/train_qwen3_32b_head_opd_dp4_mp2.py --auto-launch \
  --student-path "$STUDENT" \
  --teacher-path "$TEACHER" \
  --anno-json "$ANNO" \
  --video-dir "$VIDEOS" \
  --output-dir "$OUT" --gpus-per-worker 2 \
  --token-local-gate --boundary-epsilon 0 \
  --lr 2e-5 --max-samples 2000 --max-attempts 10000 --grad-accum 4 \
  "${extra[@]}" "$@"


