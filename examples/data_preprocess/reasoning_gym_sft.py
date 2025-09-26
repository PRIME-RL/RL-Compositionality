from datasets import load_dataset, Dataset
import pandas as pd
from verl.utils.reward_score.synthetic import compute_score
from argparse import ArgumentParser
import os
from collections import defaultdict


parser = ArgumentParser()
parser.add_argument('--gen_path', type=str, default="/fs-computility/prime/shared/chenweize/results/synthetic_string_manipulation_single_difficult/llama_3.1_8b_base_gen_train.parquet")
parser.add_argument('--data_path', type=str, default="data/synthetic_data/synthetic_string_manipulation_single_difficult/forward_train.parquet")
parser.add_argument('--save_path', type=str, default="data/string_manipulation_sft_difficult")
parser.add_argument('--max_correct_ratio', type=float, default=1.0)
args = parser.parse_args()

def filter_incorrect(example):
    new_responses = []
    for response in example['responses']:
        if compute_score(response, example['reward_model']['ground_truth'], example['data_source'])[1] == 1:
            new_responses.append(response)
    if len(new_responses) >= args.max_correct_ratio * len(example['responses']):
        new_responses = []
    return {
        "responses": new_responses,
    }

# dataset = pd.read_parquet(args.gen_path, engine="fastparquet")
# dataset = Dataset.from_pandas(dataset)
all_gen_path = []
if os.path.isdir(args.gen_path):
    for file in os.listdir(args.gen_path):
        if file.endswith(".parquet"):
            all_gen_path.append(os.path.join(args.gen_path, file))
else:
    all_gen_path = [args.gen_path]

raw_dataset = load_dataset("parquet", data_files=args.data_path)['train']
new_dataset = []
func_statistics = defaultdict(lambda: 0)

for gen_path in all_gen_path:
    dataset = load_dataset("parquet", data_files=gen_path)['train']

    dataset = dataset.map(filter_incorrect, num_proc=1, remove_columns=['responses'])
    # dataset = dataset.filter(lambda x: len(x['responses']) > 0)

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
                    "style": data['reward_model']['style'],
                    "ground_truth": data['reward_model']['ground_truth'],
                },
                "extra_info": {
                    'split': data['extra_info']['split'],
                    'index': data['extra_info']['index'],
                },
            }
            # func_statistics[prompt.split('return ')[-1].split('(')[0]] += 1
            
            new_dataset.append(new_data)

# print(func_statistics)
new_dataset = Dataset.from_list(new_dataset)
# new_dataset = new_dataset.train_test_split(test_size=1024, seed=42)
print(new_dataset[0]['prompt'])
print(f"Train size: {len(new_dataset)}")
new_dataset.to_parquet(f"{args.save_path}/train.parquet")
# new_dataset['test'].to_parquet(f"{args.save_path}/test.parquet")
