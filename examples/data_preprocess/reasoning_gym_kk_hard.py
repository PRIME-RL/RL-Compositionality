import reasoning_gym
from datasets import Dataset
import json
from tqdm import tqdm


for people in [3, 4, 5]:
    dataset = reasoning_gym.create_dataset('knights_knaves',
                                        n_people=people,
                                        depth_constraint=2,
                                        width_constraint=2,
                                        seed=0,
                                        size=512)
    new_dataset = []
    for data in tqdm(dataset):
        prompt = data['question'].split('So who is a knight and who')[0].strip()
        prompt = prompt + " At the end of your answer, list the identity of each person one by one, for example: <answer> ...(brief explanation) (1) Zoey is a knight\n(2) Oliver is a knight\n(3)... </answer>"
        sample = {
            "data_source": f"reasoning-gym-kk-depth{people}",
            "prompt": [{
                "role": "user",
                "content": prompt
            }],
            "ability": "reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps({
                    k: 'knight' if answer else 'knave' for k, answer in zip(data['metadata']['names'], data['metadata']['solution'])
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
    new_dataset.to_parquet(f"data/reasoning_gym/kk_hard/train_{people}.parquet")
