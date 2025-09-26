import reasoning_gym
from datasets import Dataset
import json
from tqdm import tqdm

TEMPLATE = """Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 * 4 </answer>."""

for number in [3, 4, 5]:
    dataset = reasoning_gym.create_dataset('countdown',
                                        min_numbers=number,
                                        max_numbers=number,
                                        min_value=1,
                                        max_value=100,
                                        min_target=1,
                                        max_target=999,
                                        operators=("+", "-", "*", "/"),
                                        shuffle=True,
                                        seed=42,
                                        size=512)
    new_dataset = []
    for data in tqdm(dataset):
        sample = {
            "data_source": f"reasoning-gym-countdown-depth{number}",
            "prompt": [{
                "role": "user",
                "content": TEMPLATE.format(numbers=str(data['metadata']['numbers']), target=data['metadata']['target'])
            }],
            "ability": "reasoning",
            "reward_model": {
                "style":
                    "rule",
                "ground_truth":
                    json.dumps({
                        "answer": data['metadata']['target'],
                        "numbers": data['metadata']['numbers']
                    })
            },
            "extra_info": {
                "index": 0,
                "split": "dummy",
            }
        }
        new_dataset.append(sample)

    new_dataset = Dataset.from_list(new_dataset)
    print(new_dataset[0]['prompt'][0]['content'])
    new_dataset.to_parquet(f"data/reasoning_gym/countdown_hard/train_{number}number.parquet")
