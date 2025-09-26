from argparse import ArgumentParser
from datasets import Dataset
import json
import random
import string
import csv
import inspect
from math import gcd

BACKWARD_PROMPT = "You are given a code:\n\n{code}\n\nGiven the following output:\n\n{output}\n\nCan you predict a feasible input without writing any code? Please reason and put your final answer in the following json format: {{\"input\": <your input>}}, where <your input> should be a dictionary, even if the there is only one input variable, with keys strictly match the input variables' names as specified."

FORWARD_PROMPT = "You are given a code:\n\n{code}\n\nCan you predict the output of `main_solution(\"{input}\")` without writing any code? Please reason and put your final answer in the following json format: {{\"output\": <your output>}}, where <your output> should be the final string."


# --------------------------------------------------
# Base Deterministic Operators
# --------------------------------------------------
def deterministic_shuffle(s):
    """Reorder characters using a fixed multiplier permutation."""
    L = len(s)
    if L == 0:
        return s
    multiplier = 3
    while gcd(multiplier, L) != 1:
        multiplier += 2
    return ''.join(s[(i * multiplier) % L] for i in range(L))


def repeat_str(s, n):
    """Repeat the string s exactly n times."""
    return s * n


def remove_vowels(s):
    """Remove vowels from the string."""
    vowels = 'aeiouAEIOU'
    return ''.join(ch for ch in s if ch not in vowels)


def sort_chars(s):
    """Sort the characters in the string."""
    return ''.join(sorted(s))


def reverse_words(s):
    """Reverse the order of words in the string."""
    words = s.split()
    return ' '.join(reversed(words))


def add_prefix(s, pre):
    """Add a fixed prefix to the string."""
    return pre + s


def add_suffix(s, suf):
    """Add a fixed suffix to the string."""
    return s + suf


def interlace_str(s1, s2):
    """Interlace two strings character by character (iterative)."""
    result = []
    len1, len2 = len(s1), len(s2)
    for i in range(max(len1, len2)):
        if i < len1:
            result.append(s1[i])
        if i < len2:
            result.append(s2[i])
    return ''.join(result)


# --------------------------------------------------
# Additional Operators Using Loops and Recursion
# --------------------------------------------------
def rotate_str(s, n):
    """Rotate the string s by n positions using slicing."""
    if not s:
        return s
    n = n % len(s)
    return s[n:] + s[:n]


def mirror_str(s):
    """Append the reversed string to the original."""
    return s + s[::-1]


def alternate_case(s):
    """Alternate the case of characters (even-index lower, odd-index upper)."""
    return ''.join(ch.lower() if i % 2 == 0 else ch.upper() for i, ch in enumerate(s))


def shift_chars(s, shift):
    """
    Shift alphabetical characters by a fixed amount (wrapping around).
    Non-letters remain unchanged.
    """

    def shift_char(ch):
        if 'a' <= ch <= 'z':
            return chr((ord(ch) - ord('a') + shift) % 26 + ord('a'))
        elif 'A' <= ch <= 'Z':
            return chr((ord(ch) - ord('A') + shift) % 26 + ord('A'))
        return ch

    return ''.join(shift_char(ch) for ch in s)


def vowel_to_number(s):
    """Replace vowels with numbers: a/A->1, e/E->2, i/I->3, o/O->4, u/U->5."""
    mapping = {'a': '1', 'e': '2', 'i': '3', 'o': '4', 'u': '5', 'A': '1', 'E': '2', 'I': '3', 'O': '4', 'U': '5'}
    return ''.join(mapping.get(ch, ch) for ch in s)


def insert_separator(s, sep):
    """Insert a fixed separator between every two characters."""
    return sep.join(s)


def duplicate_every_char(s):
    """Duplicate every character in the string."""
    return ''.join(ch * 2 for ch in s)


def fancy_brackets(s):
    """Enclose each character in fancy brackets."""
    return ''.join("«" + ch + "»" for ch in s)


def compress_repeats(s):
    """Remove adjacent duplicate characters (compress repeats)."""
    if not s:
        return s
    result = [s[0]]
    for ch in s[1:]:
        if ch != result[-1]:
            result.append(ch)
    return ''.join(result)


def recursive_reverse(s):
    """Recursively reverse the string."""
    if s == "":
        return s
    return recursive_reverse(s[1:]) + s[0]


def loop_concat(s, n):
    """Concatenate s with itself n times using a loop."""
    result = ""
    for _ in range(n):
        result += s
    return result


def while_rotate(s, n):
    """Rotate the string using a while loop (n times)."""
    count = 0
    while count < n and s:
        s = s[1:] + s[0]
        count += 1
    return s


def recursive_interlace(s1, s2):
    """Recursively interlace two strings character by character."""
    if not s1 or not s2:
        return s1 + s2
    return s1[0] + s2[0] + recursive_interlace(s1[1:], s2[1:])


def loop_filter_nonalpha(s):
    """Remove non-alphabetic characters using an explicit loop."""
    result = ""
    for ch in s:
        if ch.isalpha():
            result += ch
    return result


# --------------------------------------------------
# New Advanced Operators with Backtracking/Verification
# --------------------------------------------------
def verify_even_length(s):
    """
    Verification operator: if the length of s is even, return s;
    otherwise remove the last character.
    """
    return s if len(s) % 2 == 0 else s[:-1]


def backchain_add_digit(s, depth):
    """
    Backtracking operator: deterministically transform s so it contains a digit.
    Applies a fixed sequence of transformations recursively.
    """

    def has_digit(t):
        return any(ch.isdigit() for ch in t)

    transformations = [
        lambda t: t + "1",
        lambda t: "2" + t,
        lambda t: t.replace("a", "3"),
        lambda t: t[::-1],
    ]

    def helper(t, d):
        if has_digit(t):
            return t
        if d == 0:
            return None
        for trans in transformations:
            new_t = trans(t)
            res = helper(new_t, d - 1)
            if res is not None:
                return res
        return None

    result = helper(s, depth)
    return result if result is not None else s


def backchain_palindrome(s, depth):
    """
    Back chaining: try to transform s into a palindrome.
    If s is not already a palindrome and depth permits, append its reverse and try again.
    """
    if s == s[::-1]:
        return s
    if depth <= 0:
        return s
    new_s = s + s[::-1]
    return backchain_palindrome(new_s, depth - 1)


# --------------------------------------------------
# Custom Functions Dictionary for eval
# --------------------------------------------------

func_name_mapping = {
    "deterministic_shuffle": 'func_0',
    "repeat_str": 'func_1',
    "remove_vowels": 'func_2',
    "sort_chars": 'func_3',
    "reverse_words": 'func_4',
    "add_prefix": 'func_5',
    "add_suffix": 'func_6',
    "interlace_str": 'func_7',
    "rotate_str": 'func_8',
    "mirror_str": 'func_9',
    "alternate_case": 'func_10',
    "shift_chars": 'func_11',
    "vowel_to_number": 'func_12',
    "insert_separator": 'func_13',
    "duplicate_every_char": 'func_14',
    "fancy_brackets": 'func_15',
    "compress_repeats": 'func_16',
    "recursive_reverse": 'func_17',
    "loop_concat": 'func_18',
    "while_rotate": 'func_19',
    "recursive_interlace": 'func_20',
    "loop_filter_nonalpha": 'func_21',
    "verify_even_length": 'func_22',
    "backchain_add_digit": 'func_23',
    "backchain_palindrome": 'func_24',
}

eval_set = {
    "deterministic_shuffle": deterministic_shuffle,
    # "repeat_str": repeat_str,
    "remove_vowels": remove_vowels,
    # "sort_chars": sort_chars,
    # "reverse_words": reverse_words,
    # "add_prefix": add_prefix,
    "add_suffix": add_suffix,
    "interlace_str": interlace_str,
    "rotate_str": rotate_str,
    # "mirror_str": mirror_str,
    "alternate_case": alternate_case,
    # "shift_chars": shift_chars,
    "vowel_to_number": vowel_to_number,
    # "insert_separator": insert_separator,
    "duplicate_every_char": duplicate_every_char,
    # "fancy_brackets": fancy_brackets,
    "compress_repeats": compress_repeats,
    # "recursive_reverse": recursive_reverse,
    "loop_concat": loop_concat,
    # "while_rotate": while_rotate,
    # "recursive_interlace": recursive_interlace,
    "loop_filter_nonalpha": loop_filter_nonalpha,
    # "verify_even_length": verify_even_length,
    # "backchain_add_digit": backchain_add_digit,
    "backchain_palindrome": backchain_palindrome,
}

train_set = {
    # "deterministic_shuffle": deterministic_shuffle,
    "repeat_str": repeat_str,
    # "remove_vowels": remove_vowels,
    "sort_chars": sort_chars,
    "reverse_words": reverse_words,
    "add_prefix": add_prefix,
    # "add_suffix": add_suffix,
    # "interlace_str": interlace_str,
    # "rotate_str": rotate_str,
    "mirror_str": mirror_str,
    # "alternate_case": alternate_case,
    "shift_chars": shift_chars,
    # "vowel_to_number": vowel_to_number,
    "insert_separator": insert_separator,
    # "duplicate_every_char": duplicate_every_char,
    "fancy_brackets": fancy_brackets,
    # "compress_repeats": compress_repeats,
    "recursive_reverse": recursive_reverse,
    # "loop_concat": loop_concat,
    "while_rotate": while_rotate,
    "recursive_interlace": recursive_interlace,
    # "loop_filter_nonalpha": loop_filter_nonalpha,
    "verify_even_length": verify_even_length,
    "backchain_add_digit": backchain_add_digit,
    # "backchain_palindrome": backchain_palindrome,
}


# --------------------------------------------------
# Random Expression Generator (Deterministic Final Program)
# --------------------------------------------------
def random_expr(depth=3):
    """
    Recursively generate a random string expression in terms of 'x'.
    All extra parameters are resolved during generation so that the final
    expression (when turned into a lambda) is fully deterministic.
    """
    if depth == 0:
        # At depth 0, return either the variable 'x' or a constant literal.
        if random.random() < 0.5:
            return "x"
        else:
            literal = ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 6)))
            return f"'{literal}'"

    # Decide between binary and unary operations.
    if random.random() < 0.2:  # Binary branch.
        left = random_expr(depth - 1)
        right = random_expr(depth - 1)
        r = random.random()
        # if r < 0.33:
        #     return f"({left} + {right})"
        # elif r < 0.66:
        #     return f"interlace_str({left}, {right})"
        # else:
        #     return f"recursive_interlace({left}, {right})"
        if r < 0.5:
            return f"({left} + {right})"
        else:
            if "interlace_str" in custom_functions:
                return f"interlace_str({left}, {right})"
            else:
                return f"recursive_interlace({left}, {right})"
    else:
        # Unary branch.
        r = random.random()
        if r < 0.02:
            # Use a built-in string method.
            builtin_ops = ["upper", "lower", "capitalize", "swapcase"]
            op = random.choice(builtin_ops)
            sub_expr = random_expr(depth - 1)
            return f"({sub_expr}).{op}()"
        else:
            # Choose from our custom operators.
            no_param_custom = [
                "deterministic_shuffle", "remove_vowels", "sort_chars", "reverse_words", "mirror_str", "alternate_case",
                "vowel_to_number", "duplicate_every_char", "fancy_brackets", "compress_repeats", "recursive_reverse",
                "loop_filter_nonalpha", "verify_even_length"
            ]
            param_custom = [
                "repeat_str", "add_prefix", "add_suffix", "rotate_str", "shift_chars", "insert_separator",
                "while_rotate", "loop_concat", "backchain_add_digit", "backchain_palindrome"
            ]
            no_param_custom = [x for x in no_param_custom if x in custom_functions]
            param_custom = [x for x in param_custom if x in custom_functions]
            if random.random() < 0.5:
                op = random.choice(no_param_custom)
                sub_expr = random_expr(depth - 1)
                return f"{op}({sub_expr})"
            else:
                op = random.choice(param_custom)
                if op not in custom_functions:
                    breakpoint()
                sub_expr = random_expr(depth - 1)
                if op == "repeat_str":
                    n = random.randint(2, 4)
                    return f"repeat_str({sub_expr}, {n})"
                elif op == "add_prefix":
                    pre = ''.join(random.choices(string.ascii_lowercase, k=random.randint(2, 4)))
                    return f"add_prefix({sub_expr}, '{pre}')"
                elif op == "add_suffix":
                    suf = ''.join(random.choices(string.ascii_lowercase, k=random.randint(2, 4)))
                    return f"add_suffix({sub_expr}, '{suf}')"
                elif op == "rotate_str":
                    n = random.randint(1, 3)
                    return f"rotate_str({sub_expr}, {n})"
                elif op == "shift_chars":
                    shift_val = random.randint(1, 5)
                    return f"shift_chars({sub_expr}, {shift_val})"
                elif op == "insert_separator":
                    sep = random.choice(['-', '_', '|'])
                    return f"insert_separator({sub_expr}, '{sep}')"
                elif op == "while_rotate":
                    n = random.randint(1, 3)
                    return f"while_rotate({sub_expr}, {n})"
                elif op == "loop_concat":
                    n = random.randint(2, 4)
                    return f"loop_concat({sub_expr}, {n})"
                elif op == "backchain_add_digit":
                    d = random.randint(1, 3)
                    return f"backchain_add_digit({sub_expr}, {d})"
                elif op == "backchain_palindrome":
                    d = random.randint(1, 3)
                    return f"backchain_palindrome({sub_expr}, {d})"
    return "x"  # Fallback (should not be reached)


# --------------------------------------------------
# Helper: Generate Full Code for Test-time Execution
# --------------------------------------------------
def generate_full_code(expr):
    """
    Scan the generated expression for custom function names and concatenate
    their source codes. Then append a lambda definition that produces the final function.
    """
    used_funcs = set()
    for fname in custom_functions.keys():
        if fname in expr:
            used_funcs.add(fname)
    code_parts = []
    for fname in used_funcs:
        try:
            code_parts.append(inspect.getsource(custom_functions[fname]))
        except Exception:
            pass
    # Combine all definitions and add the lambda definition.
    # full_code = "\n\n".join(code_parts)
    full_code = f"def main_solution(x):\n    return {expr}"
    for func_name, mapped_name in func_name_mapping.items():
        full_code = full_code.replace(func_name, mapped_name)
    return full_code


# --------------------------------------------------
# Feasible Input Generator (for Testing)
# --------------------------------------------------
def generate_feasible_input(func, attempts=100, min_len=3, max_len=10):
    """
    Generate a random input string that, when passed to func, produces a valid string.
    """
    for _ in range(attempts):
        length = random.randint(min_len, max_len)
        test_input = ''.join(random.choices(string.ascii_lowercase, k=length))
        try:
            result = func(test_input)
            if isinstance(result, str):
                return test_input
        except Exception:
            continue
    return None


# --------------------------------------------------
# Function Generator: Produces a Deterministic Program
# --------------------------------------------------
def generate_random_string_function(max_depth=3):
    """
    Generates a random string-processing function (as a lambda in terms of 'x')
    with a randomly generated expression that is fully deterministic at runtime.
    
    Returns:
      expr: The string expression of the function.
      func: The lambda function.
      feasible_input: A test input string that produces an output.
    """
    expr = random_expr(max_depth)
    func_str = f"lambda x: {expr}"
    try:
        func = eval(func_str, custom_functions)
    except Exception as e:
        raise ValueError(f"Error constructing function from expression: {expr}\n{e}")

    feasible_input = generate_feasible_input(func)
    return expr, func, feasible_input


# --------------------------------------------------
# Main Execution: Generate and Save 10k Data per Depth
# --------------------------------------------------
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--local_dir', default="data/synthetic_data/synthetic_string_manipulation_composition_8level_incomplete_train_set_fix_fakename")
    args = parser.parse_args()

    # depths = [1, 2, 3, 4]
    depths = [1, 2, 3, 4, 5, 6, 7, 8]
    # depths = [1, 2]
    # depths = [1]
    # total_samples_per_depth = 10256
    # total_samples_per_depth = 102048  # Number of samples to generate for each depth
    # total_samples_per_depth = 51024
    total_samples_per_depth = 12756

    for set_idx, custom_functions in enumerate([train_set, eval_set]):
        forward_data = []
        backward_data = []

        generated = set()
        random.seed(42)
        if set_idx == 0:
            depths = [1, 2]
            total_samples_per_depth = 250000
        elif set_idx == 1:
            depths = [1, 2, 3, 4, 5, 6, 7, 8]
            total_samples_per_depth = 256
        for depth in depths:
            for i, data in enumerate([forward_data]):
                if i == 1 and depth == 1:
                    continue
                count = 0
                print(f"Generating data for depth {depth} ...")
                while count < total_samples_per_depth:
                    try:
                        expr, func, feasible_input = generate_random_string_function(max_depth=depth)
                    except Exception as e:
                        continue  # Skip if an error occurs during generation.
                    if feasible_input is None:
                        continue  # Skip if no feasible input was found.
                    try:
                        output = func(feasible_input)
                    except Exception:
                        continue  # If evaluation fails, skip this sample.

                    if expr in generated:
                        continue
                    generated.add(expr)

                    # Create the full executable code that contains all function definitions and the lambda.
                    full_code = generate_full_code(expr)
                    if i == 0:
                        sample = {
                            "data_source": f"codeio-forward-incomplete-depth{depth}",
                            "prompt": [{
                                "role": "user",
                                "content": FORWARD_PROMPT.format(code=full_code, input=feasible_input)
                            }],
                            "ability": "reasoning",
                            "reward_model": {
                                "style":
                                    "rule",
                                "ground_truth":
                                    json.dumps({
                                        "ref_input": {"x": feasible_input},
                                        "ref_output": output,
                                        "ref_code": full_code,
                                        "funcname": "main_solution",
                                    })
                            },
                            "extra_info": {
                                "index": 0,
                                "split": "dummy",
                                "depth": depth
                            }
                        }
                    else:
                        sample = {
                            "data_source": f"codeio-backward-incomplete-depth{depth}",
                            "prompt": [{
                                "role": "user",
                                "content": BACKWARD_PROMPT.format(code=full_code, output=output)
                            }],
                            "ability": "reasoning",
                            "reward_model": {
                                "style":
                                    "rule",
                                "ground_truth":
                                    json.dumps({
                                        "ref_input": {"x": feasible_input},
                                        "ref_output": output,
                                        "ref_code": full_code,
                                        "funcname": "main_solution",
                                    })
                            },
                            "extra_info": {
                                "index": 0,
                                "split": "dummy",
                                "depth": depth
                            }
                        }
                    data.append(sample)
                    count += 1
                    if count % 1000 == 0:
                        print(f"  Depth {depth}: Generated {count} samples.")

        dataset = Dataset.from_list(forward_data)
        # dataset = dataset.train_test_split(test_size=2048, seed=42)
        if set_idx == 0:
            dataset.to_parquet(f"{args.local_dir}/forward_train.parquet")
        elif set_idx == 1:
            dataset.to_parquet(f"{args.local_dir}/forward_test.parquet")
        print(
            f"Saved {len(dataset)} samples to {args.local_dir}.")

    # dataset = Dataset.from_list(backward_data)
    # dataset = dataset.train_test_split(test_size=1024, seed=42)
    # dataset['train'].to_parquet(f"{args.local_dir}/backward_train.parquet")
    # dataset['test'].to_parquet(f"{args.local_dir}/backward_test.parquet")
    # dataset['test'].select(range(100)).to_parquet(f"{args.local_dir}/backward_debug.parquet")
    # print(
    #     f"Saved {len(dataset['train'])} training samples and {len(dataset['test'])} test samples to {args.local_dir}.")