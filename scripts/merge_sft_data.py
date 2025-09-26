from argparse import ArgumentParser
from datasets import load_dataset

def main(args):
    dataset = load_dataset('parquet', data_files=args.data)['train']
    dataset.to_parquet(args.output_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, required=True, nargs='+')
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()

    main(args)