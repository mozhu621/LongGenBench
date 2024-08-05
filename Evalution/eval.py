import json
import re
import time
import pandas as pd
from vllm import LLM, SamplingParams

# 读取JSON文件
def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# 写入JSON文件
def write_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# 解析output_blocks中特定类型的条目
def parse_blocks(output_blocks, type):
    type_to_block = {}
    pattern = rf"{type} (\d+)"  # 假设类型后的数字仍有用，例如标识ID或序号
    for block in output_blocks:
        match = re.search(pattern, block)
        if match:
            identifier = int(match.group(1))  # 获取类型后的数字
            type_to_block[identifier] = block
    return type_to_block

# 生成检查内容的prompt
def create_prompts(checks, type_to_block):
    prompts = []
    identifiers = []
    for identifier, event_desc in checks.items():
        identifier = int(identifier)  # 确保转换为整数
        if identifier in type_to_block:
            prompts.append(type_to_block[identifier] + f" Does this description include the {event_desc}? Please answer with 'yes' or 'no' only.")
            identifiers.append(identifier)
    return prompts, identifiers

# 定义评估准确性的函数
def evaluate_accuracy(prompts, llm, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    results = []
    for output in outputs:
        response = output.outputs[0].text.strip().lower()
        result = 'yes' if 'yes' in response else 'no'
        results.append(result)
    return results

# 保存准确率到CSV文件
def save_accuracy_to_csv(file_path, model_name, completion_rate, acc_once, acc_range, acc_periodic):
    df = pd.DataFrame({
        'Model': [model_name],
        'Completion Rate': [completion_rate],
        'Accuracy Once': [acc_once],
        'Accuracy Range': [acc_range],
        'Accuracy Periodic': [acc_periodic],
        'Average Accuracy': [(acc_once + acc_range + acc_periodic) / 3]
    })
    
    try:
        existing_df = pd.read_csv(file_path)
        existing_df = existing_df[existing_df['Model'] != model_name]  # 删除相同模型名称的行
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        pass
    
    df.to_csv(file_path, index=False)

# 计算完成度
def calculate_completion_rate(type_to_block, total_number):
    identifiers = set(type_to_block.keys())
    expected_identifiers = set(range(1, total_number + 1))
    missing_identifiers = expected_identifiers - identifiers
    completion_rate = (len(expected_identifiers) - len(missing_identifiers)) / len(expected_identifiers)
    return completion_rate * 100



