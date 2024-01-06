#! /usr/bin/env bash

set -ex

PRE_SEQ_LEN=256
LR=2e-2
NUM_GPUS=1
MAX_SOURCE_LEN=1280
MAX_TARGET_LEN=4
DEV_BATCH_SIZE=4
GRAD_ACCUMULARION_STEPS=8
MAX_STEP=1500
SAVE_INTERVAL=500

DATESTR=`date +%Y%m%d`

BASE_MODEL_PATH=/opt/data/private/LLM2/ChatGLM3/chatglm3-6b
OUTPUT_DIR=output/ptuing-chatglm3-6b-pt-${DATESTR}-${PRE_SEQ_LEN}-${LR}

mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune.py \
    --train_format input-output \
    --train_file ../data/train.json \
    --preprocessing_num_workers 8 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --per_device_train_batch_size $DEV_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --max_steps $MAX_STEP \
    --logging_steps 100 \
    --save_steps $SAVE_INTERVAL \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN 2>&1 | tee ${OUTPUT_DIR}/train.log
