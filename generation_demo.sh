#!/bin/bash

# set -x

json_file="eval/SpatialGenEval_T2I_Prompts.jsonl"
model_name="Qwen/Qwen-Image"                 # e.g., Qwen/Qwen-Image
save_folder="<YOUR_GENERATED_IMAGE_FOLDER>"  # e.g., ./images/Qwen-Image
TOTAL_GPUS=8                                 # i.e., The expected used GPUs

for i in $(seq 0 ${TOTAL_GPUS})
do
    CUDA_VISIBLE_DEVICES=$i \
    python scripts/generation_demo.py \
        --model_name=${model_name} \
        --json_file=${json_file} \
        --save_folder=${save_folder} \
        --total_gpus=${TOTAL_GPUS} \
        --gpu_id=$i &
done
wait