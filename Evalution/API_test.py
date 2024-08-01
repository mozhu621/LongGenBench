import os
import json
from together import Together

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

def main():
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        print("API key for Together is not set in the environment variables.")
        return

    client = Together(api_key=api_key)
    input_file = '/home/yuhao/THREADING-THE-NEEDLE/Dataset/prompts_weekly_diary.json'
    inputs = load_inputs(input_file)
    
    results = []
    for input_data in inputs:
        print(input_data['prompt'])
        stream = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[{"role": "user", "content": input_data['prompt']}],
            max_tokens=2000,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>"],
            stream=True
        )
        
        output = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            print(content, end="", flush=True)
            output += content

        output_blocks = process_output(output)
        results.append(output_blocks)
    
    output_file = 'results.json'
    process_and_save_results(inputs, results, output_file)
    print("\n数据已保存到results.json文件中")

if __name__ == "__main__":
    main()
