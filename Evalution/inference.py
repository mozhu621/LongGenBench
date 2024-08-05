import argparse
import time
import json
from vllm import LLM, SamplingParams
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Run LLM with command line arguments.')
    parser.add_argument('--model', type=str, required=True, help='Model type to use.')
    parser.add_argument('--max_length', type=int, default=8000, help='Maximum length of generation.')
    parser.add_argument('--gpu_count', type=int, default=1, help='Number of GPUs to use.')
    return parser.parse_args()

# Processing output with a delimiter
def process_output(output: str) -> list:
    return output.split('###')

# Saving data to JSON format
def save_to_json(data: list, filename: str) -> None:
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
            "output_blocks": result_blocks
        })
    save_to_json(combined, filename)

args = parse_args()

input_file = '/home/yuhao/THREADING-THE-NEEDLE/Dataset/Dataset.json'
inputs = load_inputs(input_file)

sampling_params = SamplingParams(temperature=0.95, top_p=0.95, max_tokens=args.max_length, seed=42)

prompts = [input_data['prompt'] for input_data in inputs]

# Setting up the LLM with the specified number of GPUs and model
torch.cuda.set_device(args.gpu_count)  # This line sets the GPU device to use
llm = LLM(model=args.model, tensor_parallel_size=args.gpu_count)

start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
inference_time = time.time() - start_time
print(f"Inference time: {inference_time:.2f} seconds")

results = [process_output(output.outputs[0].text) for output in outputs]

output_file = f"{args.model}_maxlen{args.max_length}.json"
process_and_save_results(inputs, results, output_file)
print(f"\nSaved result to {output_file}")
