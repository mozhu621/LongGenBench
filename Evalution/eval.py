import json
import re
import time
from vllm import LLM, SamplingParams
import torch
import json

# 读取JSON文件
import json
import re

# 读取JSON文件
def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# 解析output_blocks中的周数

def parse_weeks(output_blocks, type):
    week_to_block = {}
    pattern = rf"{type} (\d+)"
    for block in output_blocks:
        match = re.search(pattern, block)
        if match:
            week_number = int(match.group(1))
            week_to_block[week_number] = block
    return week_to_block


# 生成检查内容的prompt
def create_prompts(week_checks, week_to_block):
    prompts = []
    for week_num, event_desc in week_checks.items():
        week_num = int(week_num)
        if week_num in week_to_block:
            prompts.append(week_to_block[week_num] + f"Does this description include the {event_desc}? Please answer with 'yes' or 'no' only.")
    return prompts


# 主函数
file_path = "/home/yuhao/THREADING-THE-NEEDLE/Evalution/results.json"
datas = read_json(file_path)
prompts_once= []
prompts_range = []
prompts_periodic = []
for data in datas:
    week_to_block = parse_weeks(data['output_blocks'], data['type'])
    # 生成once, range, periodic的prompts
    prompts_once.extend(create_prompts(data['checks_once'], week_to_block))
    prompts_range.extend(create_prompts(data['checks_range'], week_to_block))
    prompts_periodic.extend(create_prompts(data['checks_periodic'], week_to_block))




# Define the sampling parameters
sampling_params = SamplingParams(temperature=0.95, top_p=0.95, max_tokens=50, seed=42)

# Example lists of prompts


# Record the start time
start_time = time.time()

# Initialize the LLM with the specified model and configuration
llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", tensor_parallel_size=2)

def evaluate_accuracy(prompts, llm, sampling_params):
    # Generate responses using the LLM
    outputs = llm.generate(prompts, sampling_params)
    total = len(outputs)
    correct = sum([1 for output in outputs if 'yes' in output.outputs[0].text])
    return correct / total if total > 0 else 0

# Evaluate the accuracy for each set of prompts
acc_once = evaluate_accuracy(prompts_once, llm, sampling_params)
acc_range = evaluate_accuracy(prompts_range, llm, sampling_params)
acc_periodic = evaluate_accuracy(prompts_periodic, llm, sampling_params)

# Print the accuracy results
print("Accuracy for once:", acc_once)
print("Accuracy for range:", acc_range)
print("Accuracy for periodic:", acc_periodic)

# Print the elapsed time
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

