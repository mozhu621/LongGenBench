#!/bin/bash

# 定义 inference.py 需要的参数
MODEL_TYPE="Qwen/Qwen2-7B-Instruct"
MODEL_NAME=$(basename $MODEL_TYPE)
MAX_LENGTH=16000
NUM_GPUS=2
OUTPUT_DIR="./results"
OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_maxlen${MAX_LENGTH}.json"
export CUDA_VISIBLE_DEVICES=0,1
# 确保输出目录存在


# 运行 inference.py
python inference.py --model $MODEL_TYPE --max_length $MAX_LENGTH --gpu $NUM_GPUS --output_file $OUTPUT_FILE
# 定义 eval.py 需要的参数
CSV_PATH="/home/yuhao/THREADING-THE-NEEDLE/Evalution/results/accuracy_results.csv"

# 运行 eval.py
python eval.py --data $OUTPUT_FILE --csv $CSV_PATH --gpu $NUM_GPUS
