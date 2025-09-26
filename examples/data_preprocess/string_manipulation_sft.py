from datasets import load_dataset, Dataset
import pandas as pd
from verl.utils.reward_score.codeio import compute_score
from argparse import ArgumentParser
import os
from collections import defaultdict
from transformers import AutoTokenizer
from collections import Counter


parser = ArgumentParser()
parser.add_argument('--gen_path', type=str, default="/fs-computility/prime/shared/chenweize/results/synthetic_string_manipulation_single_difficult/llama_3.1_8b_base_gen_train.parquet")
parser.add_argument('--data_path', type=str, default="data/synthetic_data/synthetic_string_manipulation_single_difficult/forward_train.parquet")
parser.add_argument('--save_path', type=str, default="data/string_manipulation_sft_difficult")
parser.add_argument('--val_size', type=int, default=1024)
parser.add_argument('--max_correct_ratio', type=float, default=1.0)
parser.add_argument('--max_length', type=int, default=None)
parser.add_argument('--tokenizer', type=str, default=None)
parser.add_argument('--no_remove_context', action='store_true')
parser.add_argument('--remove_overlong', action='store_true')
args = parser.parse_args()

if args.max_length:
    assert args.tokenizer is not None
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
else:
    tokenizer = None

def filter_incorrect(example):
    new_responses = []
    for response in example['responses']:
        if compute_score(response, example['reward_model']['ground_truth'], example['data_source'])[1] == 1:
            if args.max_length:
                if len(tokenizer(response)['input_ids']) <= args.max_length:
                    new_responses.append(response)
            else:
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

    dataset = dataset.map(filter_incorrect, num_proc=4, remove_columns=['responses'])
    # dataset = dataset.filter(lambda x: len(x['responses']) > 0)

    for data, raw_data in zip(dataset, raw_dataset):
        for response in data['responses']:
            if args.remove_overlong:
                if not response.strip().endswith('<|eot_id|>'):
                    continue
            # new_data = {k: v for k, v in data.items() if k != 'responses'}
            prompt = raw_data['prompt'][0]['content']
            if not args.no_remove_context:
                prompt = prompt[prompt.find("def main_solution(x):"):]
                prompt = f"You are given a code:\n\n{prompt}"
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
            func_statistics[prompt.split('return ')[-1].split('(')[0]] += 1
            
            new_dataset.append(new_data)
print(func_statistics)
new_dataset = Dataset.from_list(new_dataset)
data_source = new_dataset['data_source']
print(Counter(data_source).items())
if args.val_size:
    new_dataset = new_dataset.train_test_split(test_size=args.val_size, seed=42)
    train_dataset, test_dataset = new_dataset['train'], new_dataset['test']
    print(f"Train size: {len(new_dataset['train'])}, Test size: {len(new_dataset['test'])}")
    new_dataset['train'].to_parquet(f"{args.save_path}/train.parquet")
    new_dataset['test'].to_parquet(f"{args.save_path}/test.parquet")
else:
    new_dataset.to_parquet(f"{args.save_path}/train.parquet")
    print(f"Saved {len(new_dataset)} data.")