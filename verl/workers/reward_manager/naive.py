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

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import random

class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

    def detect_repetition_with_hash(self, text, window_size=10):
        """
        Use hashing to efficiently detect repetitions
        """
        words = text.split()
        if len(words) <= window_size:
            return 0.0
        
        seen_hashes = set()
        repetitions = 0
        
        for i in range(len(words) - window_size + 1):
            # Get window and its hash
            window = tuple(words[i:i+window_size])
            window_hash = hash(window)
            
            # Check if we've seen this hash before
            if window_hash in seen_hashes:
                repetitions += 1
            else:
                seen_hashes.add(window_hash)
        
        # # Normalize repetition score
        # repetition_penalty_type = self.config.verifier.get('repetition_penalty','binary')
        # if repetition_penalty_type=="ratio":
        #     repetition_score = - 2*repetitions / (len(words) - window_size + 1) if len(words) > window_size else 0
        # elif repetition_penalty_type=="binary":
        #     # if repetitions >= 3 and repetitions < 5:
        #     #     repetition_score = -1 
        #     # elif repetitions >= 5 and repetitions < 10:
        #     #     repetition_score = -2
        #     # elif repetitions >= 10:
        #     #     repetition_score = -3
        #     # else:
        #     #     repetition_score = 0
        #     repetition_score = 0 if repetitions < 5 else -1
        # else:
        #     raise NotImplementedError
        repetition_score = 0 if repetitions < 5 else -1
        
        return repetition_score

    def verify(self, data):
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            scores.append(score)
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        acc_tensor = torch.zeros(len(data), dtype=torch.float32, device=data.batch['responses'].device)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score, acc = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            repetition = self.detect_repetition_with_hash(response_str)
            reward_tensor[i, valid_response_length - 1] = score #+ repetition
            acc_tensor[i] = acc

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            do_print = random.randint(1, 1024) == 1
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                if do_print:
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)
                    print("[ground_truth]", ground_truth)
                    print("[score]", score)

        return reward_tensor, acc_tensor
