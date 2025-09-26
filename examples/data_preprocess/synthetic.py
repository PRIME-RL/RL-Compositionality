# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import json
import math
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
import sympy as sp

from verl.utils.hdfs_io import copy, makedirs
import argparse

np.random.seed(0)

def countdown_processor(example):
    expr = example["reward_model"]["ground_truth"]
    expr = re.sub(r"(\d)!", r"factorial(\1)", expr)
    expr = re.sub(r"√(\d+)", r"sqrt(\1)", expr)
    answer = sp.sympify(expr, evaluate=True)
    assert answer.is_integer
    answer = str(answer)

    problem = example["prompt"][0]["content"].rstrip("\n Output: \n").strip()
    problem = problem.split('\n')
    # problem = "Create a mathematical expression using the four given numbers to reach the target result. Each number appears only once. You are allowed to use +, -, *, /. Parentheses are also allowed.\n" + problem[-2] + '\n' + problem[-1].strip()
    # problem = (
    #     problem
    #     + "\nProvide the final expression within <answer> tags. For example: <answer> 1 + 2 * 3 - 4 </answer>\n"
    # )
    numbers = re.findall(r'(?<!\^)(?<!\d)(\d+)', example['reward_model']['ground_truth'])
    assert len(numbers) == 4
    numbers_int = [int(x) for x in numbers]
    problem = f"Using the numbers {numbers_int}, create an equation that equals {answer}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 * 4 </answer>."
    data = {
        "data_source": "synthetic_countdown",
        "prompt": [
            {
                "role": "user",
                "content": problem,
            }
        ],
        "ability": "countdown",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps({"answer": answer, "numbers": numbers}, ensure_ascii=False),
        },
        # "extra_info": {
        #     "difficulty": example["extra_info"]["difficulty"],
        # },
        "difficulty": int(example["extra_info"]["difficulty"]),
    }
    return data


def kk_processor(example):
    problem = (example["prompt"][0]["content"].rstrip(
        "So who is a knight and who is a knave?\nPlease list all combinations of Knight.").strip())
    # problem = (
    #     problem
    #     + ' You should provide the final answer at the end of your response in the following json format wrapped with "```json":\n\n```json\n{\n  "knight": ["A","B"],\n  "knave": ["C"]\n}\n```'
    # )
    problem = problem + ' At the end of your answer, list the identity of each person one by one, for example: <anwer> ...(brief explanation) (1) Zoey is a knight\n(2) Oliver is a knight\n(3)... </answer>'
    # answer = {"knight": [], "knave": []}
    # for name, solution in zip(example["extra_info"]["names"], example["extra_info"]["solution"]):
    #     if solution:
    #         answer["knight"].append(name)
    #     else:
    #         answer["knave"].append(name)
    answer = {}
    for name, solution in zip(example["extra_info"]["names"], example["extra_info"]["solution"]):
        if solution:
            answer[name] = "knight"
        else:
            answer[name] = "knave"
    data = {
        "data_source": "synthetic_kk",
        "prompt": [{
            "role": "user",
            "content": problem,
        }],
        "ability": "kk",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(answer, ensure_ascii=False),
        },
        # "extra_info": {
        #     "difficulty": example["extra_info"]["difficulty"],
        # },
        "difficulty": int(example["extra_info"]["difficulty"]),
    }
    return data


def sudoku_processor(example):
    problem = example["prompt"][0]["content"].strip().rstrip("Please provide the complete solution.").strip()
    problem = (
        problem
        + '\nThink step by step. At the end of the response, wrap the completed puzzle with ```, e.g.,\n```\n[\n [num, num, ...]\n, ...\n]\n```'
    )
    data = {
        "data_source": "synthetic_sudoku",
        "prompt": [
            {
                "role": "user",
                "content": problem,
            }
        ],
        "ability": "sudoku",
        "reward_model": {
            "style": "rule",
            "ground_truth": example['prompt'][0]['content'].split('\n\n')[1],
        },
        # "extra_info": {
        #     "difficulty": example["extra_info"]["difficulty"],
        # },
        "difficulty": int(example["extra_info"]["difficulty"]),
    }
    return data


def zebra_processor(example):
    problem = "\n".join(example["prompt"][0]["content"].strip().split('\n')[:-1]).lstrip("Puzzle Components \n").strip()
    complete_solution = json.loads(example['extra_info']['solution'])
    candidates = list(complete_solution.keys())
    # item_num = np.random.randint(min(3, len(candidates)), min(len(candidates) + 1, 6))
    # sampled_candidates = np.random.choice(candidates, item_num, replace=False)
    sampled_candidates_text = ', '.join(candidates)
    solution = {k: complete_solution[k] for k in candidates}
    # problem = problem + f"\n\nWhere are {sampled_candidates_text}? You should provide the final answer at the end of your response in the following json format wrapped with \"```json\":\n\n```json\n{{\n  \"Alice\": 1,\n  \"Bob\": 2\n}}\n```"
    problem = problem + f'\n\nWhere are {sampled_candidates_text}?  At the end of your answer, list their positions one by one, for example: <anwer> ...(brief explanation)\n(1) The Chocolate smoothie is in nest 1\n(2) The Fierce Traptor is in nest 2\n(3) ... </answer>'
    data = {
        "data_source": "synthetic_zebra",
        "prompt": [
            {
                "role": "user",
                "content": problem,
            }
        ],
        "ability": "zebra",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(solution, ensure_ascii=False),
        },
        # "extra_info": {
        #     "difficulty": example["extra_info"]["difficulty"],
        # },
        "difficulty": int(example["extra_info"]["difficulty"]),
    }
    return data


def stratified_train_test_split(dataset, test_size=2048, shuffle=True, seed=42):
    """
    Stratified train test split according to data_source and difficulty
    :param dataset: the dataset to split
    :param test_size: the number of samples in the test set
    :return: a tuple of train and test datasets
    """
    train_dataset = {column: [] for column in dataset.column_names}
    test_dataset = {column: [] for column in dataset.column_names}
    count = {}
    for example in tqdm(dataset, desc="Counting samples", total=len(dataset)):
        key = f"{example['data_source']}_{example['extra_info']['difficulty']}"
        count[key] = count.get(key, 0) + 1

    total = sum(count.values())
    for key in count:
        count[key] = max(math.ceil(count[key] / total * test_size), 1)

    print(count)

    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    test_remain_count = deepcopy(count)
    test_cnt = 0
    for example in tqdm(dataset, desc="Splitting samples", total=len(dataset)):
        key = f"{example['data_source']}_{example['extra_info']['difficulty']}"
        if test_remain_count[key] > 0:
            test_remain_count[key] -= 1
            test_cnt += 1
            for column in dataset.column_names:
                test_dataset[column].append(example[column])
        else:
            for column in dataset.column_names:
                train_dataset[column].append(example[column])

    if test_cnt > test_size:
        test_to_train_index = np.random.choice(len(test_dataset["data_source"]), test_cnt - test_size, replace=False)
        for index in sorted(test_to_train_index, reverse=True):
            for column in dataset.column_names:
                train_dataset[column].append(test_dataset[column].pop(index))

    return Dataset.from_dict(train_dataset), Dataset.from_dict(test_dataset)


PROCESSOR_MAP = {
    "24point_like_data": countdown_processor,
    "knight_knave_puzzle": kk_processor,
    "sudoku_4x4": sudoku_processor,
    "sudoku_9x9": sudoku_processor,
    "logic_puzzle_challenge": zebra_processor,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/synthetic_data")
    parser.add_argument("--data_dir", default="/data1/chenweize/datasets/synthetic_data/")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    datasets: list[Dataset] = []
    for file in os.listdir(args.data_dir):
        if file.endswith(".parquet"):
            dataset = load_dataset("parquet", data_files=os.path.join(args.data_dir, file))["train"]
            # dataset = dataset  #.shuffle(42).select(range(min(50000, len(dataset))))
            datasets.append(dataset)
    sudoku_merge = []
    for i in range(len(datasets)):
        print(f"processing {datasets[i][0]['data_source']}")
        datasets[i] = datasets[i].map(
            PROCESSOR_MAP[datasets[i][0]["data_source"]],
            remove_columns=datasets[i].column_names,
            num_proc=1 if args.debug else 32,
        )
        datasets[i] = datasets[i].map(
            lambda x: {"extra_info": {
                "difficulty": x["difficulty"]
            }},
            remove_columns=["difficulty"],
            num_proc=1 if args.debug else 32,
        )
        if "sudoku" in datasets[i][0]['data_source']:
            sudoku_merge.append((i, datasets[i]))
    subsets = {}
    for dataset in datasets:
        train_dataset, test_dataset = stratified_train_test_split(
            dataset,
            test_size=1024,
            shuffle=True,
            seed=42,
        )
        subsets[dataset[0]["data_source"]] = {
            "train": train_dataset,
            "test": test_dataset,
        }

    sudoku_dataset = concatenate_datasets([x[1] for x in sudoku_merge])  #.shuffle(42).select(range(50000))
    indexes = sorted([x[0] for x in sudoku_merge], reverse=True)
    for index in indexes:
        datasets.pop(index)
    datasets.append(sudoku_dataset)

    for i in range(len(datasets)):
        if datasets[i][0]['data_source'] == "synthetic_countdown":
            datasets[i] = datasets[i].filter(
                lambda x: x["extra_info"]["difficulty"] <= 2,
                num_proc=1 if args.debug else 32,
            )
        datasets[i] = datasets[i].shuffle(42).select(range(min(50000, len(datasets[i]))))

    dataset = concatenate_datasets(datasets)
    train_dataset, test_dataset = stratified_train_test_split(
        dataset,
        test_size=4096,
        shuffle=True,
        seed=42,
    )

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    # test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    # test_dataset.select(range(100)).to_parquet(os.path.join(local_dir, "debug.parquet"))

    for subset in subsets:
        if subset == "synthetic_countdown":
            subsets[subset]["train"].to_parquet(os.path.join(local_dir, subset, "train.parquet"))
            subsets[subset]["test"].to_parquet(os.path.join(local_dir, subset, "test.parquet"))
            subsets[subset]["test"].select(range(100)).to_parquet(os.path.join(local_dir, subset, "debug.parquet"))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
