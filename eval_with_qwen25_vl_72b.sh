#!/bin/bash

# set -x

export OPENAI_API_KEY=None

# ------------------------------------------------------
# Stage 0: Initialize Job Arguments 
# ------------------------------------------------------
echo "====== Stage 0: Initialize Job Arguments ======"
INPUT_JSON="./eval/SpatialGenEval_T2I_Prompts.jsonl"  # Input prompts file: <Do not need to change> 
IMAGE_PATH=$1                                         # Input images path: <Replace with your path of generated images, involving 1230 images>
OUTPUT_JSON=$2                                        # Output eval results path <Replace with your path >
EVAL_MODEL_PATH=$3                                    # Eval model path: <Replace with your path of Qwen2.5-VL-72B-Instruct>
IMAGE_PATHES=(${IMAGE_PATH})  

echo "INPUT_JSON: ${INPUT_JSON}"
echo "IMAGE_PATH: ${IMAGE_PATH}"
echo "OUTPUT_JSON: ${OUTPUT_JSON}"
echo "EVAL_MODEL_PATH: ${EVAL_MODEL_PATH}"

# ------------------------------------------------------
# Stage 1: MLLM Initialization 
# ------------------------------------------------------
echo "====== Stage 1: MLLM Initialization ======"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p logs
log_file="./logs/vllm_server_qwen25_vl_72b.log"
API_NAME="Qwen2.5-VL-72B-Instruct"
nohup vllm serve "${EVAL_MODEL_PATH}" \
    --port 8001 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --served_model_name "${API_NAME}" \
    --tensor-parallel-size 8 \
    > ${log_file} 2>&1 &

echo "Waiting for vllm server to start..."
SLEEP_TIME=120
while ! curl -s http://0.0.0.0:8001 > /dev/null; do
    echo "Waiting vllm starting···, the detailed logs are shown in ${log_file}. This process will cost a few minutes."
    sleep ${SLEEP_TIME}
    SLEEP_TIME=$(( (SLEEP_TIME - 20) > 20 ? (SLEEP_TIME - 20) : 20 ))
done

rollout_pid=$!
echo "vllm service started and is ready to accept requests."
nvidia-smi

# ------------------------------------------------------
# Stage 2: MLLM as an understanding model for inference
# ------------------------------------------------------
echo "====== Stage 2: VLM as an understanding model for inference ======"
ROLLOUT=5                           # Rollout: <do not need to change>
COUNT=4                             # Minimun accurate count: <do not need to change>
MAX_WORKERS=50                      # Maximum workers (more workers to speed up): <you can change it to suit for your machine>, <8*H20, Qwen2.5-VL-72B, num_worker=50, will cost about 40 minutes for evaluation>

STAGE1_PYTHON_SCRIPT="scripts/spatialgeneval_stage1_eval.py"
# Scene List
SCENES=(
    "Forest" "Mountain" "Beach" "Desert" "Underwater" "Cityspace" "Village" 
    "Portrait" "Human Activities" "Sports" 
    "Office" "Living Room" "Kitchen" "Classroom"
    "Park" "Zoo" 
    "Airport" "Railway Station" "Art Gallery" "Shopping Mall" "Library" "Cafe" 
    "Advertisement Design" "Cartoon Design" "Story Design"
)
# Eval
for image_pth in "${IMAGE_PATHES[@]}"; do
    echo "=========================================================="
    echo ">>>>> Running evaluation for image_pth: \"$image_pth\""
    echo "=========================================================="

    begin_time=$(date +%s)

    for scene in "${SCENES[@]}"; do
        echo "=========================================================="
        echo ">>>>> Running evaluation for scene: \"$scene\" from \"$image_pth\""  
        echo "=========================================================="

        python "$STAGE1_PYTHON_SCRIPT" \
            --input_json "$INPUT_JSON" \
            --output_json "$OUTPUT_JSON" \
            --image_pth "$image_pth" \
            --api_name "$API_NAME" \
            --base_url "http://0.0.0.0:8001/v1" \
            --scene "$scene" \
            --rollout "$ROLLOUT" \
            --count "$COUNT" \
            --max_workers "$MAX_WORKERS"

        # Check 
        if [ $? -ne 0 ]; then
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            echo "Stopping script. Error occurred while processing scene: \"$scene\" from \"$image_pth\"."
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            exit 1
        fi
    done

    end_time=$(date +%s)
    total_seconds=$((end_time - begin_time))
    hours=$((total_seconds / 3600))
    minutes=$(((total_seconds % 3600) / 60))
    seconds=$((total_seconds % 60))
    formatted_time=$(printf "%02d:%02d:%02d" $hours $minutes $seconds)

    echo "=========================================================="
    echo ">>>>> All scenes processed successfully. From: \"$image_pth\" "
    echo ">>>>> Total time: $formatted_time"
    echo "=========================================================="
done

# ------------------------------------------------------
# Stage 3: Calculate the accuracy for SpatialGenEval
# ------------------------------------------------------
STAGE2_PYTHON_SCRIPT="scripts/spatialgeneval_stage2_acc.py"
python "$STAGE2_PYTHON_SCRIPT" "$OUTPUT_JSON"

echo "Killing vllm server..."
kill $rollout_pid
