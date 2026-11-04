#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
OUT="${OUT:-../Lkmllm_data/checkpoints/qwen3_boundary_opd_$(date +%Y%m%d_%H%M%S)}"
torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_qwen3_32b_head_opd_dp4_mp2.py \
  --student-path "${STUDENT:-../shared_models/Qwen3-VL-8B-Instruct}" \
  --teacher-path "${TEACHER:-../shared_models/Video-opd-teacher-Qwen3-VL-32B-GRPO}" \
  --anno-json "${ANNO:-../Lkmllm_data/datasets/Train/timelens-100k/timelens-100k.jsonl}" \
  --video-dir "${VIDEOS:-../Lkmllm_data/datasets/Train/timelens-100k}" \
  --output-dir "$OUT" --gpus-per-worker 2 \
  --boundary-selective --boundary-epsilon 0 \
  --lr 2e-5 --max-samples 1000 --max-attempts 10000 --grad-accum 4 \
  --offload-saved-tensors "$@"
