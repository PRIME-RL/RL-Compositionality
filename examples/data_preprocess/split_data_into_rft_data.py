from datasets import load_dataset
from argparse import ArgumentParser
import os

def main(args):
    dataset = load_dataset('parquet', data_files=args.data_path)['train']
    dataset = dataset.shuffle(42)

    for iter_id in range(args.num_iter):
        start_idx = iter_id * args.example_per_iter
        end_idx = (iter_id + 1) * args.example_per_iter
        iter_dataset = dataset.select(range(start_idx, end_idx))
        iter_dataset.to_parquet(os.path.join(args.save_path, str(iter_id) + '.parquet'))
        print('====================================')
        print(f'Example from Iter {iter_id}')
        print(iter_dataset[0]['prompt'][0]['content']) 

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--save-path', type=str, required=True)
    parser.add_argument('--example-per-iter', type=int, default=2000)
    parser.add_argument('--num-iter', type=int, default=10)
    args = parser.parse_args()

    main(args)