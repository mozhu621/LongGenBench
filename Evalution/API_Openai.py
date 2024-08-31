import argparse
import time
import json
import os
import openai

def parse_args():
    parser = argparse.ArgumentParser(description='Generate text using OpenAI API.')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Model to use for generation.')
    parser.add_argument('--max_tokens', type=int, default=1600, help='Maximum number of tokens to generate.')
    parser.add_argument('--output_file', type=str, required=True, help='Output file path.')
    return parser.parse_args()

def process_output(output):
    blocks = output.split('###')
    word_count = len(output.split())
    return {"blocks": blocks, "word_count": word_count}

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def save_to_json(data, filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_inputs(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_and_save_results(inputs, results, filename):
    combined = []
    for input_data, result_data in zip(inputs, results):
        combined.append({
            "input": input_data["prompt"],
            "output_blocks": result_data["blocks"],
            "word_count": result_data["word_count"]
        })
    save_to_json(combined, filename)

def inference_API(prompts, args):
    openai.api_key = 'sk-Ipwx4rA5gulbqSr8DDpg6F_ECMtF0-VeLdwJFjVAsaT3BlbkFJgz9MsPuGmVW37nSBvYpRwR0gR7_TkWI8YUg6ujpcAA'  # Use your actual API key
    results = []
    print("start")
    for prompt in prompts:
        try:
            response = openai.ChatCompletion.create(
                model=args.model,
                messages=[{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
                max_tokens=args.max_tokens,
                temperature=0.7,
                seed=6211027,
                stop=["*** finished"]
            )
            print(response.choices[0].message.content.strip())
            results.append(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"Failed to process prompt {prompt}: {str(e)}")
            results.append("")
    return results

args = parse_args()

input_file = '/home/yuhao/THREADING-THE-NEEDLE/Dataset/Dataset_short.json'  # Adjust the path to your input file
inputs = load_inputs(input_file)
prompts = [input_data['prompt'] for input_data in inputs]
prompts = prompts[:2]
start_time = time.time()
outputs = inference_API(prompts, args)
inference_time = time.time() - start_time
print(f"Inference time: {inference_time:.2f} seconds")

results = [process_output(output) for output in outputs]
process_and_save_results(inputs, results, args.output_file)
print(f"\nSaved result to {args.output_file}")
