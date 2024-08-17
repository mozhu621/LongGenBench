import argparse
import time
import json

import torch
import os
from together import Together
# meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
# python inference_API.py --model modeltogethercomputer/Llama-2-7B-32K-Instruct --max_length 32000 --output_file Llama-3.1-70B-Instruct-Turbo_max_length_short.json
def parse_args():
    parser = argparse.ArgumentParser(description='Run LLM with command line arguments.')
    parser.add_argument('--model', type=str, required=True, help='Model type to use.')
    parser.add_argument('--max_length', type=int, default=8000, help='Maximum length of generation.')
    #parser.add_argument('--gpu', type=int, default=1, help='Number of GPUs to use.')
    parser.add_argument('--output_file', type=str, required=True, help='Output file path.')
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
    directory = os.path.dirname(filename)
    if not directory:  # Check if directory is empty
        directory = '.'  # Set current directory if no directory is provided
    os.makedirs(directory, exist_ok=True)
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


def inference_API(inputs, args):
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        print("API key for Together is not set in the environment variables.")
        return []

    client = Together(api_key=api_key)
    results = []
    for input_data in inputs:
        # print(f"Generating for prompt: {input_data['prompt']}")
        stream = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": input_data}],
            max_tokens=args.max_length,
            temperature=0.95,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>"],
            stream=True
        )
        
        output = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            #print(content, end="", flush=True)
            output += content
        
        results.append(output)
    return results

args = parse_args()

input_file = '/home/yuhao/THREADING-THE-NEEDLE/Dataset/Dataset_short.json'
inputs = load_inputs(input_file)

prompts = [input_data['prompt'] for input_data in inputs]

# Setting up the LLM with the specified number of GPUs and model

start_time = time.time()
outputs = inference_API(prompts, args)
inference_time = time.time() - start_time
print(f"Inference time: {inference_time:.2f} seconds")

results = [process_output(output) for output in outputs]

process_and_save_results(inputs, results, args.output_file)
print(f"\nSaved result to {args.output_file}")
