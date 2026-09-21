#!/bin/bash
# Merge LoRA adapter into base model, then extract teacher features for distillation

set -e
ENV=/home/flyingbird/anaconda3/envs/llama_factory/bin
PYTHON=/home/flyingbird/anaconda3/envs/qwen_vl_vadb/bin/python3
BASE=/home/flyingbird/Work/AutoDriver/FSDrive-main

echo "=== Step 1: Merge LoRA adapter into base model ==="
unset ALL_PROXY all_proxy HTTPS_PROXY HTTP_PROXY http_proxy https_proxy
export PATH="$ENV:$PATH"

cd $BASE/LLaMA-Factory
llamafactory-cli export \
    --model_name_or_path $BASE/model/FSDrive_pretrain \
    --adapter_name_or_path $BASE/model/lora_traj_5ep \
    --template qwen2_vl \
    --finetuning_type lora \
    --export_dir $BASE/model/lora_traj_5ep_merged \
    --export_size 2 \
    --export_legacy_format false \
    --trust_remote_code true

echo "=== Step 2: Extract train features ==="
cd $BASE
$PYTHON create_data/extract_teacher_features.py \
    --model_path model/lora_traj_5ep_merged \
    --split train \
    --output_dir /media/flyingbird/07419DF8D71B0526/Dataset/nuScenes/teacher_features_lora5ep \
    --resume

echo "=== Step 3: Extract val features ==="
$PYTHON create_data/extract_teacher_features.py \
    --model_path model/lora_traj_5ep_merged \
    --split val \
    --output_dir /media/flyingbird/07419DF8D71B0526/Dataset/nuScenes/teacher_features_lora5ep \
    --resume

echo "=== Done ==="
