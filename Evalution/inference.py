import argparse
import time
import json
from vllm import LLM, SamplingParams
import torch
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Run LLM with command line arguments.')
    parser.add_argument('--model', type=str, required=True, help='Model type to use.')
    parser.add_argument('--max_length', type=int, default=8000, help='Maximum length of generation.')
    parser.add_argument('--gpu', type=int, default=1, help='Number of GPUs to use.')
    parser.add_argument('--output_file', type=str, required=True, help='Output file path.')
    parser.add_argument('--input_file', type=str, required=True, help='input file path.')
  
    args = parser.parse_args()
    return args

# Process output to split blocks and count words
def process_output(output: str) -> dict:
    blocks = output.split('###')
    word_count = len(output.split())
    return {"blocks": blocks, "word_count": word_count}

def read_json(file_path):
    with open(file_path, 'r',encoding='utf-8') as file:
        return json.load(file)
    
def write_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def save_to_json(data: list, filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_inputs(filename: str) -> list:
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Combine inputs, results and word counts and save them
def process_and_save_results(inputs: list, results: list, filename: str) -> None:
    combined = []
    for input_data, result_data in zip(inputs, results):
        combined.append({
            "input": input_data["prompt"],
            "checks_once": input_data["checks_once"],
            "checks_range": input_data["checks_range"],
            "checks_periodic": input_data["checks_periodic"],
            "type": input_data["type"],
            "number": input_data['number'],
            "output_blocks": result_data["blocks"],
            "word_count": result_data["word_count"]  # Adding word count here
        })
    save_to_json(combined, filename)

args = parse_args()

#input_file = '/home/yuhao/THREADING-THE-NEEDLE/Dataset/Dataset_short.json'
inputs = load_inputs(args.input_file)

# sampling_params = SamplingParams(temperature=0.95, top_p=0.95, max_tokens=args.max_length, seed=42, repetition_penalty = 1.005)
sampling_params = SamplingParams(temperature=0.95, top_p=0.95, max_tokens=args.max_length, seed=42, stop = '*** finished')

prompts = [input_data['prompt'] for input_data in inputs]

# Setting up the LLM with the specified number of GPUs and model
llm = LLM(model=args.model, tensor_parallel_size=args.gpu, gpu_memory_utilization=0.95)

start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
inference_time = time.time() - start_time
print(f"Inference time: {inference_time:.2f} seconds")

results = [process_output(output.outputs[0].text) for output in outputs]

process_and_save_results(inputs, results, args.output_file)
print(f"\nSaved result to {args.output_file}")
