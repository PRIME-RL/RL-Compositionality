from datasets import load_dataset, Dataset
import random

dataset = load_dataset("parquet", data_files="data/string_manipulation_sft_difficult_2/train.parquet")

all_ops = [
    # Binary ops
    "interlace_str", "recursive_interlace",
    # Built-in unary ops
    "upper", "lower", "capitalize", "swapcase",
    # Custom unary ops (no params)
    "deterministic_shuffle", "remove_vowels", "sort_chars", "reverse_words", "mirror_str",
    "alternate_case", "vowel_to_number", "duplicate_every_char", "fancy_brackets",
    "compress_repeats", "recursive_reverse", "loop_filter_nonalpha", "verify_even_length",
    # Custom unary ops (with params)
    "repeat_str", "add_prefix", "add_suffix", "rotate_str", "shift_chars",
    "insert_separator", "while_rotate", "loop_concat", "backchain_add_digit",
    "backchain_palindrome"
]

op_acc = {}
for data in dataset['train']:
    for op in all_ops:
        if op + '(' in data['prompt']:
            if op not in op_acc:
                op_acc[op] = []
            op_acc[op].append(data)

print({k: len(v) for k, v in op_acc.items()})

final_dataset = []

for op, data_list in op_acc.items():
    random.shuffle(data_list)
    final_dataset += data_list[:4096]  # Take 1024 samples for each operation

final_dataset = Dataset.from_list(final_dataset)
final_dataset.to_parquet("data/string_manipulation_sft_difficult_2/train_balanced.parquet")