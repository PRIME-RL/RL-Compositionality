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

import time
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError

import torch
from tqdm import tqdm

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


def _worker_function(evaluation_func, task_desc, completion, reference, task_extra_info):
    """
    Wrapper function to call the evaluation function.
    This will be run in a separate process.
    """
    return evaluation_func(data_source=task_desc, solution_str=completion, ground_truth=reference, extra_info=task_extra_info)


def parallel_compute_score_sync(evaluation_func,
                                completions,
                                references,
                                tasks_descriptions,
                                extra_info=None,
                                num_processes=64,
                                timeout_per_task=300.):
    """
    Computes scores in parallel using ProcessPoolExecutor.
    Returns a list of raw results from the evaluation function.
    """
    num_items = len(completions)
    results = [(0, 0)] * num_items

    if extra_info is None:
        extra_info = [None] * num_items

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all tasks to the executor
        futures_map = {
            executor.submit(_worker_function, evaluation_func, tasks_descriptions[i], completions[i], references[i], extra_info[i]): i
            for i in range(num_items)
        }

        print(f"Submitted {len(futures_map)} tasks to {num_processes} processes. Waiting for results...")
        
        # tqdm can be used here to show progress
        for future in tqdm(futures_map, total=num_items, desc="Computing rewards"):
            idx = futures_map[future]
            try:
                results[idx] = future.result(timeout=timeout_per_task)
            except FuturesTimeoutError:
                print(f"Timeout ({timeout_per_task}s) for task at index {idx}. Result is set to None.")
                # The result at results[idx] is already None, so no action needed.
            except Exception as e:
                print(f"An error occurred while processing task at index {idx}: {e}. Halting batch.")
                # To prevent partial results, we re-raise the exception to stop processing this batch.
                executor.shutdown(wait=False, cancel_futures=True)
                raise
    
    return results


class MPRewardManager:
    """
    A multi-processing Reward Manager that computes rewards in parallel.

    This manager orchestrates the scoring of model responses by distributing the
    computation across multiple processes. It handles decoding, parallel execution,
    result aggregation, and applies additional penalties like repetition checks.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        num_processes=32,
        task_timeout=10.,
        penalize_overlong=False,
        max_response_length=None
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.num_processes = num_processes
        self.task_timeout = task_timeout
        self.penalize_overlong = penalize_overlong
        self.max_response_length = max_response_length

    def detect_repetition_with_hash(self, text, window_size=10):
        """
        Use hashing to efficiently detect repetitions and return a penalty score.
        """
        words = text.split()
        if len(words) <= window_size:
            return 0.0
        
        seen_hashes = set()
        repetitions = 0
        
        for i in range(len(words) - window_size + 1):
            window = tuple(words[i:i+window_size])
            window_hash = hash(window)
            
            if window_hash in seen_hashes:
                repetitions += 1
            else:
                seen_hashes.add(window_hash)
        
        # Simple binary penalty: -1 if 5 or more repetitions are found.
        repetition_score = 0 if repetitions < 5 else -1
        return repetition_score

    def __call__(self, data: DataProto):
        """
        Computes rewards for the given data batch in parallel.

        This method performs the following steps:
        1. Decodes prompts and responses from token IDs to strings.
        2. Gathers necessary metadata (ground truth, data source, etc.).
        3. Calls `parallel_compute_score_sync` to compute scores across multiple processes.
        4. Parses the results, handling various return formats (dicts, floats).
        5. Calculates a repetition penalty for each response.
        6. Combines the base score and repetition penalty for the final reward.
        7. Assigns the final reward to the last token of each response in a tensor.
        8. Populates an 'acc' tensor with the base scores (for accuracy tracking).
        9. Prints a few samples for qualitative examination.
        """
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        batch_size = len(data)
        device = data.batch['responses'].device

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        acc_tensor = torch.zeros(batch_size, dtype=torch.float32, device=device)
        
        # 1. Decode all necessary strings and gather metadata upfront
        prompts_str = self.tokenizer.batch_decode(data.batch['prompts'], skip_special_tokens=True)
        responses_str = self.tokenizer.batch_decode(data.batch['responses'], skip_special_tokens=True)
        ground_truths = [item.non_tensor_batch['reward_model']['ground_truth'] for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_info_in = data.non_tensor_batch.get('extra_info', None)

        # 2. Compute scores in parallel
        try:
            computed_results = parallel_compute_score_sync(
                self.compute_score, responses_str, ground_truths, data_sources,
                extra_info=extra_info_in, num_processes=self.num_processes,
                timeout_per_task=self.task_timeout
            )
        except Exception as e:
            print(f"Batch reward computation failed: {e}. Setting all scores for this batch to 0.")
            computed_results = [(0, 0)] * batch_size

        if len(computed_results) != batch_size:
            print(f"Warning: Score list length mismatch. Expected {batch_size}, got {len(computed_results)}. Resetting scores to 0.")
            computed_results = [(0, 0)] * batch_size
        
        # 3. Process results, calculate final rewards, and print samples
        already_print_data_sources = defaultdict(int)
        prompt_length = data.batch['prompts'].shape[-1]
        valid_response_lengths = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)

        for i in range(batch_size):
            result = computed_results[i]
            score, acc = result
            
            response_len = valid_response_lengths[i].item()
            if self.penalize_overlong:
                if response_len == self.max_response_length and data.batch['responses'][i][-1] != self.tokenizer.eos_token_id:
                    score = -1
                    acc = 0

            # --- Calculate repetition penalty ---
            # repetition_penalty = self.detect_repetition_with_hash(responses_str[i])
            
            # --- Final reward is the combination of score and penalties ---
            final_reward = score #+ repetition_penalty

            # --- Assign rewards and accuracy scores to tensors ---
            acc_tensor[i] = acc  # 'acc' tensor stores the score before penalties

            if response_len > 0:
                reward_tensor[i, response_len - 1] = final_reward
            elif final_reward != 0.0:
                print(f"Warning: Non-zero reward {final_reward} for empty response at index {i}. Reward not assigned.")

            # --- Logic for printing examination samples ---
            data_source = data_sources[i]
            if already_print_data_sources[data_source] < self.num_examine:
                # Print a fraction of samples to avoid console spam
                if random.randint(1, 10) == 1:
                    already_print_data_sources[data_source] += 1
                    print("-" * 50)
                    print(f"[Examining Sample] Data Source: {data_source}, Batch Index: {i}")
                    print(f"[Prompt]: {prompts_str[i]}")
                    print(f"[Response]: {responses_str[i]}")
                    print(f"[Ground Truth]: {ground_truths[i]}")
                    print(f"[Base Score]: {score}")
                    print(f"[Acc]: {acc}")
                    # print(f"[Repetition Penalty]: {repetition_penalty}")
                    print(f"[Final Reward]: {final_reward}")
                    print("-" * 50)

        return reward_tensor, acc_tensor
