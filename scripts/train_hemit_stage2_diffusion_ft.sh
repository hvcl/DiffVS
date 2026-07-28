#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

DATASET_ROOT="${DATASET_ROOT:-/path/to/HEMIT}"
STAGE1_CHECKPOINT_DIR="${STAGE1_CHECKPOINT_DIR:-./outputs/hemit_stage1_marigold/stage1-checkpoint-epoch-100}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/hemit_stage2_diffusion_ft}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-stabilityai/stable-diffusion-2-1-base}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

accelerate launch --num_processes "${NUM_PROCESSES}" src/diffvs/train_stage2_diffusion_ft.py \
  --dataset hemit \
  --dataset_root "${DATASET_ROOT}" \
  --split train \
  --markers HEMIT \
  --image_size 512 \
  --pretrained_model "${PRETRAINED_MODEL}" \
  --stage1_checkpoint_dir "${STAGE1_CHECKPOINT_DIR}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --gradient_accumulation_steps 1 \
  --mixed_precision "${MIXED_PRECISION}" \
  --save_every 5 \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
