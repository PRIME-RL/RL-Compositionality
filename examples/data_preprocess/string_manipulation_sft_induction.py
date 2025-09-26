from datasets import load_dataset, Dataset
import pandas as pd
from verl.utils.reward_score.codeio import compute_score
from argparse import ArgumentParser
import os
import json

parser = ArgumentParser()
parser.add_argument('--gen_path', type=str, default="/fs-computility/prime/shared/chenweize/results/synthetic_string_manipulation_induction_1level/llama_3.1_8b_gen_train.parquet")
parser.add_argument('--data_path', type=str, default="data/synthetic_data/synthetic_string_manipulation_induction_1level_with_testcase/train.parquet")
parser.add_argument('--save_path', type=str, default="data/string_manipulation_induction_sft")
args = parser.parse_args()

def filter_incorrect(example):
    new_responses = []
    if json.loads(example['reward_model.ground_truth'])['functions'] == '':
        return {
            'responses': []
        }
    for response in example['responses']:
        if compute_score(response, example['reward_model.ground_truth'], example['data_source'])[1] == 1:
            new_responses.append(response)
    return {
        "responses": new_responses,
    }

dataset = pd.read_parquet(args.gen_path, engine="fastparquet")
dataset = Dataset.from_pandas(dataset)
raw_dataset = load_dataset("parquet", data_files=args.data_path)['train']
dataset = dataset.select(range(len(dataset) - len(raw_dataset), len(dataset)))
os.environ['SANDBOX_FUSION_ENDPOINT'] = "http://172.31.32.65:8080"

dataset = dataset.map(filter_incorrect, num_proc=96, remove_columns=['responses'])
# dataset = dataset.filter(lambda x: len(x['responses']) > 0)
new_dataset = []

for data, raw_data in zip(dataset, raw_dataset):
    for response in data['responses']:
        # new_data = {k: v for k, v in data.items() if k != 'responses'}
        prompt = raw_data['prompt'][0]['content']
        # prompt = prompt[prompt.find("def main_solution(x):"):]
        # prompt = f"You are given a code:\n\n{prompt}"
        new_data = {
            "data_source": data['data_source'],
            "prompt": prompt,
            "response": response,
            "ability": data['ability'],
            "reward_model": {
                "style": data['reward_model.style'],
                "ground_truth": data['reward_model.ground_truth'],
            },
            "extra_info": {
                'split': data['extra_info.split'],
                'index': data['extra_info.index'],
            },
        }
        
        new_dataset.append(new_data)

new_dataset = Dataset.from_list(new_dataset)
new_dataset = new_dataset.train_test_split(test_size=1024, seed=42)
print(f"Train size: {len(new_dataset['train'])}, Test size: {len(new_dataset['test'])}")
new_dataset['train'].to_parquet(f"{args.save_path}/train.parquet")
new_dataset['test'].to_parquet(f"{args.save_path}/test.parquet")
