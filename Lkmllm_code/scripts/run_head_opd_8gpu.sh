#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${OUT:-../Lkmllm_data/checkpoints/qwen3_boundary_opd_$(date +%Y%m%d_%H%M%S)}"
echo "Training output: $OUT"
exec python -u scripts/train_qwen3_32b_head_opd_dp4_mp2.py --auto-launch \
  --student-path "${STUDENT:-../shared_models/Qwen3-VL-8B-Instruct}" \
  --teacher-path "${TEACHER:-../shared_models/Video-opd-teacher-Qwen3-VL-32B-GRPO}" \
  --anno-json "${ANNO:-../Lkmllm_data/datasets/Train/timelens-100k/timelens-100k.jsonl}" \
  --video-dir "${VIDEOS:-../Lkmllm_data/datasets/Train/timelens-100k}" \
  --output-dir "$OUT" --gpus-per-worker 2 \
  --boundary-selective --boundary-epsilon 0 \
  --lr 2e-5 --max-samples 1000 --max-attempts 10000 --grad-accum 4 \
  --offload-saved-tensors "$@"
