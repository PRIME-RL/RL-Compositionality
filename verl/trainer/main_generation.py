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
Generate responses given a dataset of prompts
"""

import csv
import ray
import numpy as np
import hydra
import os
import json
from tabulate import tabulate
from datasets import load_dataset

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.reward_score import _default_compute_score


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    from pprint import pprint
    from omegaconf import OmegaConf

    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    if os.path.exists(config.data.output_path) and not config.data.overwrite:
        print(f"Output file {config.data.output_path} already exists. Skipping generation and proceeding to evaluation.")
        # dataset = pd.read_parquet(config.data.output_path)
        dataset = load_dataset("parquet", data_files=config.data.output_path)['train']
    else:
        local_path = copy_to_local(config.model.path)
        from verl.utils import hf_tokenizer

        tokenizer = hf_tokenizer(local_path)

        if config.rollout.temperature == 0.0:
            assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."

        # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
        # dataset = pd.read_parquet(config.data.path)
        dataset = load_dataset("parquet", data_files=config.data.path)['train']
        chat_lst = dataset[config.data.prompt_key]

        chat_lst = [chat for chat in chat_lst]

        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
        resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.init_model()

        total_samples = len(dataset)
        # real_batch_size = data.batch['input_ids'].shape[0]
        config_batch_size = config.data.batch_size
        dispatch_dp_size = wg.world_size
        num_batch = -(-total_samples // config_batch_size)
        output_lst = [[] for _ in range(config.data.n_samples)]

        for batch_idx in range(num_batch):
            print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
            batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
            if config.data.zero:
                from verl.utils.dataset.rl_dataset import ZERO_TEMPLATE

                inputs = tokenizer(
                    [ZERO_TEMPLATE.format(prompt=chat) for chat in batch_chat_lst],
                    padding=True,
                    truncation=True,
                    max_length=config.rollout.prompt_length,
                    return_tensors="pt",
                )
            else:
                inputs = tokenizer.apply_chat_template(
                    batch_chat_lst,
                    add_generation_prompt=True,
                    padding=True,
                    truncation=True,
                    max_length=config.rollout.prompt_length,
                    return_tensors="pt",
                    return_dict=True,
                    tokenize=True,
                )
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            position_ids = compute_position_id_with_mask(attention_mask)

            batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

            data = DataProto.from_dict(batch_dict)
            real_batch_size = data.batch['input_ids'].shape[0]
            if real_batch_size % dispatch_dp_size != 0:
                dummy_data_size = dispatch_dp_size - real_batch_size % dispatch_dp_size
                if dummy_data_size <= real_batch_size:
                    dummy_data = data[:dummy_data_size]
                else:
                    dummy_data = data.repeat(-(-dummy_data_size // real_batch_size))[:dummy_data_size]
                data = DataProto.concat([data, dummy_data])
                print(
                    f'real_batch_size {real_batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}, add {dummy_data_size} dummy data'
                )

            batch_size = data.batch['input_ids'].shape[0]
            assert batch_size % dispatch_dp_size == 0, f'batch_size {batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}'

            print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
            # START TO GENERATE FOR n_samples TIMES
            for i in range(config.data.n_samples):
                print(f"Generating {i+1}/{config.data.n_samples}")
                output = wg.generate_sequences(data)
                # remove dummy data
                output = output[:real_batch_size]
                output_text = tokenizer.batch_decode(output.batch['input_ids'][:, -config.rollout.response_length:],
                                                    skip_special_tokens=False)

                # remove the padding
                pad_token = tokenizer.pad_token
                output_text_unpad = []
                for text in output_text:
                    output_text_unpad.append(text.replace(pad_token, ''))

                output_dir = os.path.dirname(config.data.output_path)
                makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, f'gen_{i}.json'), 'w') as f:
                    json.dump(output_text_unpad, f)

                output_lst[i].extend(output_text_unpad)

        # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
        output_lst = np.array(output_lst, dtype=object)
        output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

        # add to the data frame
        # dataset[f'responses'] = output_lst
        dataset = dataset.add_column("responses", output_lst)

        # write to a new parquet
        output_dir = os.path.dirname(config.data.output_path)
        makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "gen_all.json"), 'w') as f:
            json.dump(output_lst, f)
        dataset.to_parquet(config.data.output_path)

    output_dir = os.path.dirname(config.data.output_path)
    # Compute evaluation metrics
    prompts = dataset[config.data.prompt_key]
    responses = dataset['responses']  # Using the generated responses
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    passes = 0
    total = len(dataset)
    total_scores = []
    
    for i in range(total):
        response_lst = responses[i]
        data_source = data_sources[i]
        prompt = prompts[i]
        reward_data = reward_model_data[i]
        ground_truth = reward_data['ground_truth']
        score_lst = []
        for r in response_lst:
            score = _default_compute_score(data_source, r, ground_truth)
            score_lst.append(score)
        max_score = np.max(score_lst)
        total_scores.append(score_lst)
        if max_score == 1:
            passes += 1

    n_samples = config.data.n_samples
    pass_at_n = passes / total
    pass_at_1 = np.mean(total_scores)

    # Save metrics to CSV
    csv_path = os.path.join(output_dir, 'pass.csv')
    
    # Prepare the row data
    # Extract the dataset name from the path
    dataset_name = os.path.basename(config.data.path)
    row_data = {
        'model_path': config.model.path,
        'dataset': dataset_name,
        'pass@1': pass_at_1,
        f'pass@{n_samples}': pass_at_n
    }

    # Check if file exists
    file_exists = os.path.isfile(csv_path)
    
    # Write to CSV
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    # Convert the row data into a list of lists format for tabulate
    table_data = [[k, v] for k, v in row_data.items()]
    
    # Print table
    print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))

# Add the select_reward_fn from main_eval.py
# def select_reward_fn(data_source):
#     if data_source == 'lighteval/MATH':
#         from verl.utils.reward_score import math
#         return math.compute_score
#     else:
#         from deepscaler.rewards.math_reward import deepscaler_reward_fn
#         return deepscaler_reward_fn


if __name__ == "__main__":
    main()
