from datasets import load_dataset

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

op_correct_cnt = {op: 0 for op in all_ops}
dataset = load_dataset("parquet", data_files="data/string_manipulation_sft/train.parquet")
for data in dataset['train']:
    for op in all_ops:
        if op + '(' in data['response']:
            op_correct_cnt[op] += 1
breakpoint()