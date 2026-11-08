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
OUT="${OUT:-$ASSET_ROOT/Lkmllm_data/checkpoints/qwen3_opd_rkl_gt244_2kx2_nvidia2_$(date +%Y%m%d_%H%M%S)}"
EVAL_OUT="${EVAL_OUT:-$ASSET_ROOT/Lkmllm_data/outputs/eval_results/$(basename "$OUT")_eval14336_dp2}"
PY="${PY:-python}"
command -v "$PY" >/dev/null || { echo "Python not found: $PY" >&2; exit 2; }

# This pipeline intentionally uses exactly two NVIDIA GPUs.
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES HSA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1
BACKEND="$("$PY" -c 'import torch; assert torch.version.cuda and not torch.version.hip and torch.cuda.is_available() and torch.cuda.device_count()==2, "Require exactly two visible NVIDIA CUDA GPUs"; print("cuda")')"

QV_ANNO="${QV_ANNO:-$ASSET_ROOT/Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights-timelens.json}"
QV_VIDEOS="${QV_VIDEOS:-$ASSET_ROOT/Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights}"
CHARADES_ANNO="${CHARADES_ANNO:-$ASSET_ROOT/Lkmllm_data/datasets/Test/Charades_sta/charades-timelens.json}"
CHARADES_VIDEOS="${CHARADES_VIDEOS:-$ASSET_ROOT/Lkmllm_data/datasets/Test/Charades_sta}"
ACTIVITYNET_ANNO="${ACTIVITYNET_ANNO:-$ASSET_ROOT/Lkmllm_data/datasets/Test/Activitynet/activitynet-timelens.json}"
ACTIVITYNET_VIDEOS="${ACTIVITYNET_VIDEOS:-$ASSET_ROOT/Lkmllm_data/datasets/Test/Activitynet}"
for file in "$ANNO" "$QV_ANNO" "$CHARADES_ANNO" "$ACTIVITYNET_ANNO"; do
  [[ -f "$file" ]] || { echo "Missing annotation: $file" >&2; exit 2; }
done
for folder in "$STUDENT" "$TEACHER" "$VIDEOS" "$QV_VIDEOS" "$CHARADES_VIDEOS" "$ACTIVITYNET_VIDEOS"; do
  [[ -d "$folder" ]] || { echo "Missing model/video directory: $folder" >&2; exit 2; }
done

extra=()
case "${CPU_OFFLOAD:-1}" in
  1) extra+=(--offload-saved-tensors) ;;
  0) ;;
  *) echo 'CPU_OFFLOAD must be 0 or 1' >&2; exit 2 ;;
esac
echo "Python: $PY; backend: $BACKEND; CPU offload: ${CPU_OFFLOAD:-1}"
echo "Asset root: ${ASSET_ROOT:-explicit paths}"
echo "Training output: $OUT"
echo 'RKL; 2000 accepted events per round x 2; each round short/medium/long=400/800/800'
"$PY" -u scripts/train_qwen3_32b_head_opd_dp4_mp2.py --auto-launch \
  --student-path "$STUDENT" \
  --teacher-path "$TEACHER" \
  --anno-json "$ANNO" \
  --video-dir "$VIDEOS" \
  --output-dir "$OUT" --gpus-per-worker 2 \
  --token-local-gate --boundary-epsilon 0 \
  --distill-kl reverse --rounds 2 --duration-ratio 2 4 4 \
  --lr 2e-5 --max-samples 2000 --max-attempts 0 --grad-accum 4 \
  --fps 2 --total-tokens 14336 --min-tokens 64 --max-frames 0 \
  --clip-total-tokens 4096 --clip-fps 4 --clip-max-frames 128 \
  --quiet --log-every 10 \
  "${extra[@]}" "$@"

# Training exports one merged student. Never evaluate partial/adapter-only output.
"$PY" -c 'import json,sys; from pathlib import Path; s=json.loads((Path(sys.argv[1])/"summary.json").read_text()); assert s["training_complete"] and s["merged"] and s["accepted_events"]==s["accepted_target"], "Incomplete training/merge; refusing evaluation"' "$OUT"

evaluate() {
  "$PY" -u scripts/qwen_baseline_eval_dp.py \
    --model-path "$OUT" --anno-json "$2" --video-dir "$3" \
    --output-dir "$EVAL_OUT/$1" --split "$1" --prompt-mode decimal_seconds \
    --fps 2 --total-tokens 14336 --min-tokens 64 --max-frames 0 \
    --max-new-tokens 1024 --max-samples 0 --decode-timeout 600
}
evaluate Qvhighlights "$QV_ANNO" "$QV_VIDEOS"
evaluate Charades_sta "$CHARADES_ANNO" "$CHARADES_VIDEOS"
evaluate Activitynet "$ACTIVITYNET_ANNO" "$ACTIVITYNET_VIDEOS"
echo "Training and all three evaluations finished: $EVAL_OUT"
