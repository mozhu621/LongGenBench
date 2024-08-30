#!/bin/bash

# 定义 inference.py 需要的参数
MODEL_TYPE="mistralai/Mistral-7B-Instruct-v0.2"
# MODEL_TYPE="Qwen/Qwen2-7B-Instruct"
# MODEL_TYPE="meta-llama/Meta-Llama-3.1-8B-Instruct"
# MODEL_TYPE="google/gemma-2-9b"
# MODEL_TYPE="In2Training/FILM-7B"
# export VLLM_ATTENTION_BACKEND=FLASHINFER



MODEL_NAME=$(basename $MODEL_TYPE)
MAX_LENGTH=32000
NUM_GPUS=4
INPUT_DIR="/home/yuhao/THREADING-THE-NEEDLE/Dataset/Dataset_long.json"
OUTPUT_DIR="./results"
OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_maxlen${MAX_LENGTH}.json"
export CUDA_VISIBLE_DEVICES=2,3,4,5
# 确保输出目录存在
# 运行 inference.py
python inference.py --model $MODEL_TYPE --max_length $MAX_LENGTH --gpu $NUM_GPUS  --input_file $INPUT_DIR  --output_file $OUTPUT_FILE 
CSV_PATH="/home/yuhao/THREADING-THE-NEEDLE/Evalution/results/accuracy_results.csv"
# 运行 eval.py
python eval.py --data $OUTPUT_FILE --csv $CSV_PATH --gpu $NUM_GPUS



MODEL_TYPE="Qwen/Qwen2-7B-Instruct"
MODEL_NAME=$(basename $MODEL_TYPE)
OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_maxlen${MAX_LENGTH}.json"

python inference.py --model $MODEL_TYPE --max_length $MAX_LENGTH --gpu $NUM_GPUS  --input_file $INPUT_DIR  --output_file $OUTPUT_FILE 
# 运行 eval.py
python eval.py --data $OUTPUT_FILE --csv $CSV_PATH --gpu $NUM_GPUS


MODEL_TYPE="meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_NAME=$(basename $MODEL_TYPE)
OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_maxlen${MAX_LENGTH}.json"

python inference.py --model $MODEL_TYPE --max_length $MAX_LENGTH --gpu $NUM_GPUS  --input_file $INPUT_DIR  --output_file $OUTPUT_FILE 
# 运行 eval.py
python eval.py --data $OUTPUT_FILE --csv $CSV_PATH --gpu $NUM_GPUS


# MODEL_TYPE="google/gemma-2-9b"
# MODEL_NAME=$(basename $MODEL_TYPE)
# OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_maxlen${MAX_LENGTH}.json"

# python inference.py --model $MODEL_TYPE --max_length $MAX_LENGTH --gpu $NUM_GPUS  --input_file $INPUT_DIR  --output_file $OUTPUT_FILE 
# # 运行 eval.py
# python eval.py --data $OUTPUT_FILE --csv $CSV_PATH --gpu $NUM_GPUS


MODEL_TYPE="In2Training/FILM-7B"
MODEL_NAME=$(basename $MODEL_TYPE)
OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_maxlen${MAX_LENGTH}.json"
python inference.py --model $MODEL_TYPE --max_length $MAX_LENGTH --gpu $NUM_GPUS  --input_file $INPUT_DIR  --output_file $OUTPUT_FILE 
# 运行 eval.py
python eval.py --data $OUTPUT_FILE --csv $CSV_PATH --gpu $NUM_GPUS


MODEL_TYPE="THUDM/LongWriter-llama3.1-8b"
MODEL_NAME=$(basename $MODEL_TYPE)
OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_maxlen${MAX_LENGTH}.json"
python inference.py --model $MODEL_TYPE --max_length $MAX_LENGTH --gpu $NUM_GPUS  --input_file $INPUT_DIR  --output_file $OUTPUT_FILE 
# 运行 eval.py
python eval.py --data $OUTPUT_FILE --csv $CSV_PATH --gpu $NUM_GPUS