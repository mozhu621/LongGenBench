import time
from vllm import LLM, SamplingParams
import torch
import json
# Sample prompts.

# Create a sampling params object.


def process_output(output: str) -> list:
    """Process the output to divide it into blocks using '###' as a delimiter."""
    return output.split('###')

def save_to_json(data: list, filename: str) -> None:
    """Save the processed data to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_inputs(filename: str) -> list:
    """Load input data from a JSON file."""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def process_and_save_results(inputs: list, results: list, filename: str) -> None:
    """Combine inputs and results and save to a JSON file."""
    combined = []
    for input_data, result_blocks in zip(inputs, results):
        combined.append({
            #"id": input_data["id"],
            "input": input_data["prompt"],
            "output_blocks": result_blocks
        })
    save_to_json(combined, filename)

input_file = '/home/yuhao/THREADING-THE-NEEDLE/Dataset/prompts_weekly_diary.json'
inputs = load_inputs(input_file)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens = 8000, seed = 42,presence_penalty = 0.2,frequency_penalty=0.2)

prompts =[]
for input_data in inputs:
    prompts.append(input_data['prompt'])

# Create an LLM.
start_time = time.time()
llm = LLM(model="Qwen/Qwen2-7B-Instruct", tensor_parallel_size=2)
#llm = LLM(model="Qwen/Qwen2-7B-Instruct")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
inference_time = time.time() - start_time
print(f"Inference time: {inference_time:.2f} seconds")
# Print the outputs.
results = []
for output in outputs:
    output_blocks = process_output(output)
    results.append(output_blocks)

output_file = 'results.json'
process_and_save_results(inputs, results, output_file)
print("\n数据已保存到results.json文件中")