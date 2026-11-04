#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
OUT="${OUT:-../Lkmllm_data/checkpoints/qwen3_headopd_32b_dp4_mp2_1k_14336_$(date +%Y%m%d_%H%M%S)}"
ARGS=(
  --student-path ../shared_models/Qwen3-VL-8B-Instruct
  --teacher-path ../shared_models/Video-opd-teacher-Qwen3-VL-32B-GRPO
  --heads-json heads/qwen3_gtmask_top10.json
  --anno-json ../Lkmllm_data/datasets/Train/timelens-100k/timelens-100k.jsonl
  --video-dir ../Lkmllm_data/datasets/Train/timelens-100k
  --output-dir "$OUT"
  --teacher-mode gt_clip --gate teacher-better
  --max-samples 1000 --max-attempts 10000 --epochs 1
  --student-total-tokens 14336 --student-min-tokens 64
  --teacher-total-tokens 14336 --teacher-min-tokens 64
  --total-tokens 14336 --min-tokens 64 --fps 2 --max-frames 0
  --pool-factor 2 --max-pool-frames 1536
  --clip-total-tokens 4096 --clip-fps 4 --clip-max-frames 128
  --clip-padding-seconds 0
  --lr 2e-5 --rank 8 --alpha 16 --grad-accum 4
  --max-new-tokens 80 --max-grad-norm 1
  --student-device 0 --student-reserve-gib 12 --teacher-reserve-gib 4
  --offload-saved-tensors --save-every-steps 25 --seed 42
)
echo "Output: $OUT"
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/train_qwen3_32b_head_opd_dp4_mp2.py "${ARGS[@]}" "$@"
