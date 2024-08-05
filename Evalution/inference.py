import argparse
import time
import json
from vllm import LLM, SamplingParams
import torch
import os
from eval import parse_blocks,create_prompts,evaluate_accuracy,save_accuracy_to_csv,calculate_completion_rate
def parse_args():
    parser = argparse.ArgumentParser(description='Run LLM with command line arguments.')
    parser.add_argument('--model', type=str, required=True, help='Model type to use.')
    parser.add_argument('--max_length', type=int, default=8000, help='Maximum length of generation.')
    parser.add_argument('--gpu', type=int, default=1, help='Number of GPUs to use.')
    return parser.parse_args()

# Processing output with a delimiter
def process_output(output: str) -> list:
    return output.split('###')

def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
def write_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Saving data to JSON format
def save_to_json(data: list, filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure directory exists
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Loading input data from a JSON file
def load_inputs(filename: str) -> list:
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Combining inputs and results and saving them
def process_and_save_results(inputs: list, results: list, filename: str) -> None:
    combined = []
    for input_data, result_blocks in zip(inputs, results):
        combined.append({
            "input": input_data["prompt"],
            "checks_once": input_data["checks_once"],
            "checks_range": input_data["checks_range"],
            "checks_periodic": input_data["checks_periodic"],
            "type":input_data["type"],
            "number":input_data['number'],
            "output_blocks": result_blocks
        })
    save_to_json(combined, filename)

args = parse_args()

input_file = '/home/yuhao/THREADING-THE-NEEDLE/Dataset/Dataset.json'
inputs = load_inputs(input_file)

sampling_params = SamplingParams(temperature=0.95, top_p=0.95, max_tokens=args.max_length, seed=42)

prompts = [input_data['prompt'] for input_data in inputs]

# Setting up the LLM with the specified number of GPUs and model
  # This line sets the GPU device to use
llm = LLM(model=args.model, tensor_parallel_size=args.gpu)

start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
inference_time = time.time() - start_time
print(f"Inference time: {inference_time:.2f} seconds")

results = [process_output(output.outputs[0].text) for output in outputs]
model_name = args.model.split('/')[-1] 
output_dir = './results'
output_file = f"{output_dir}/{model_name}_maxlen{args.max_length}.json"
process_and_save_results(inputs, results, output_file)
print(f"\nSaved result to {output_file}")

### eval

csv_file_path = "/home/yuhao/THREADING-THE-NEEDLE/Evalution/results/accuracy_results.csv"
model_name = output_file.split('/')[-1].replace('.json', '')
datas = read_json(output_file)


prompts_once = []
prompts_range = []
prompts_periodic = []
identifiers_once = []
identifiers_range = []
identifiers_periodic = []

completion_rate = 0
for data in datas:
    checks_block = parse_blocks(data['output_blocks'], data['type'])
    # 生成once, range, periodic的prompts
    p_once, ids_once = create_prompts(data['checks_once'], checks_block)
    p_range, ids_range = create_prompts(data['checks_range'], checks_block)
    p_periodic, ids_periodic = create_prompts(data['checks_periodic'], checks_block)
    
    prompts_once.extend(p_once)
    identifiers_once.extend(ids_once)
    
    prompts_range.extend(p_range)
    identifiers_range.extend(ids_range)
    
    prompts_periodic.extend(p_periodic)
    identifiers_periodic.extend(ids_periodic)

    data['count_once'] = len(ids_once)
    data['count_range'] = len(ids_range)
    data['count_periodic'] = len(ids_periodic)

    
    # 计算完成度
    completion_rate += calculate_completion_rate(checks_block, data['number'])

completion_rate /= len(datas)  # 平均完成度

# Define the sampling parameters
sampling_params = SamplingParams(temperature=0.95, top_p=0.95, max_tokens=50, seed=42)

# Record the start time
start_time = time.time()

# Initialize the LLM with the specified model and configuration
llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", tensor_parallel_size=2)

# Evaluate the accuracy for each set of prompts
results_once = evaluate_accuracy(prompts_once, llm, sampling_params)
results_range = evaluate_accuracy(prompts_range, llm, sampling_params)
results_periodic = evaluate_accuracy(prompts_periodic, llm, sampling_params)

# 计算准确率
acc_once = sum(1 for result in results_once if result == 'yes') / len(results_once) if results_once else 0
acc_range = sum(1 for result in results_range if result == 'yes') / len(results_range) if results_range else 0
acc_periodic = sum(1 for result in results_periodic if result == 'yes') / len(results_periodic) if results_periodic else 0

# 将结果添加到JSON文件中
start_index_once = 0
start_index_range = 0
start_index_periodic = 0
for data in datas:
    data['results_once'] = {str(identifiers_once[i]): results_once[i] for i in range(start_index_once, start_index_once + data['count_once'])}
    start_index_once += data['count_once']
    
    data['results_range'] = {str(identifiers_range[i]): results_range[i] for i in range(start_index_range, start_index_range + data['count_range'])}
    start_index_range += data['count_range']
    
    data['results_periodic'] = {str(identifiers_periodic[i]): results_periodic[i] for i in range(start_index_periodic, start_index_periodic + data['count_periodic'])}
    start_index_periodic += data['count_periodic']

# 写回JSON文件
write_json(output_file , datas)

# 保存准确率到CSV文件
save_accuracy_to_csv(csv_file_path, model_name, completion_rate, acc_once, acc_range, acc_periodic)

# Print the elapsed time
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

