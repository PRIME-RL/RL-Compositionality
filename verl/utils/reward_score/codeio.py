import re
import json
import math
import random

solution_prefix = """from itertools import accumulate, chain, combinations, count, permutations, product, groupby, islice, repeat
from copy import deepcopy
import signal
from string import ascii_lowercase, ascii_uppercase
from math import floor, log2, log10, sqrt, hypot, comb, gcd, ceil, inf, isqrt, lcm, factorial, dist
from collections import defaultdict, deque, Counter
from bisect import bisect, bisect_left, bisect_right, insort
from heapq import heappush, heappop, heapify, merge, heapreplace
from functools import reduce, lru_cache, cache, cmp_to_key
from random import randrange, shuffle
from operator import itemgetter, sub, or_, xor, and_
from re import search as re_search  # Assuming 're' refers to a regex search
from os.path import commonprefix
from typing import List, Tuple, Dict, Set, Optional, Union, Any, Callable, Iterable, Iterator, Generator
import copy
import datetime
import string
import math
from math import atan2, pi
import collections
import bisect
import heapq
from heapq import nlargest
import functools
import random
from random import randint
import itertools
import operator
import re
import json
import numpy as np
from math import log, prod  # 'log' and 'prod' are functions in the math module
from collections import deque, defaultdict, Counter, OrderedDict
from itertools import accumulate, permutations, combinations, product, groupby, islice, chain, repeat, zip_longest, cycle
from functools import lru_cache, reduce, partial
import sys
from itertools import pairwise"""

template_check_input = """{solution_prefix}

{refcode}

def is_close(pred, target, tol=0.001):
    if isinstance(pred, dict) and isinstance(target, dict):
        if pred.keys() != target.keys():
            return False
        return all(is_close(pred[k], target[k], tol) for k in pred)

    elif isinstance(pred, list) and isinstance(target, list):
        if len(pred) != len(target):
            return False
        return all(is_close(p, t, tol) for p, t in zip(pred, target))

    elif isinstance(pred, (int, float)) and isinstance(target, (int, float)):
        if isinstance(pred, float) or isinstance(target, float):
            # if we have non number, like nan, inf, we should not compare them
            if math.isnan(pred) or math.isnan(target) or math.isinf(pred) or math.isinf(target):
                return False
            return (abs(pred - target) <= tol * abs(target)) and (int(pred) == int(target))
        return pred == target

    else:
        return pred == target

def diy_check_input_output():
    iiiioooo = {io}

    input_xx = iiiioooo['input']  # should be a json object
    output_xx = iiiioooo['output']  # should be a json object

    warning_string = "[Mismatch] Your input is not feasible! Given the output <<<<3>>>>, your predicted input is <<<<1>>>>, which actually gets a wrong output as <<<<2>>>>"

    string_iii = json.dumps(input_xx)
    string_ooo = json.dumps(output_xx).strip()

    execed_output = None

    if not {bypass}:
        if isinstance(input_xx, dict):
            execed_output = {funcname}(**input_xx)
        else:
            execed_output = {funcname}(input_xx)
    else:
        execed_output = {funcname}(input_xx)

    string_eee = json.dumps(execed_output).strip()

    cond1 = string_ooo == string_eee
    cond2 = is_close(execed_output, output_xx)

    assert cond1 or cond2, warning_string.replace(
        "<<<<1>>>>", string_iii).replace("<<<<2>>>>", string_eee).replace("<<<<3>>>>", string_ooo)

diy_check_input_output()
"""
global do_print
original_print = print

def print(*args, **kwargs):
    if do_print:
        original_print(*args, **kwargs)


def sub_extract_last_complete_json(s):
    if '```json' not in s:
        # Stack to keep track of opening and closing braces
        stack = []
        last_json_start = None
        last_json_str = None

        for i, char in enumerate(s):
            if char == '{':
                stack.append(i)
                if last_json_start is None:
                    last_json_start = i
            elif char == '}':
                if stack:
                    start = stack.pop()
                    if not stack:
                        # Complete JSON object found
                        last_json_str = s[last_json_start:i + 1]
                        last_json_start = None
    else:
        # find the last ```json
        last_json_start = s.rfind('```json')
        last_json_end = s.find('```', last_json_start + len('```json'))
        last_json_str = s[last_json_start + 7:last_json_end].strip()

    # Load the last JSON object

    if last_json_str:
        try:
            return json.loads(last_json_str.replace("\n", ""))
        except json.JSONDecodeError:
            # replace 'False', 'True' to 'false', 'true'
            last_json_str = last_json_str.replace("False", "false").replace("True", "true").replace("None", "null")
            try:
                return json.loads(last_json_str.replace("\n", ""))
            except json.JSONDecodeError:
                pass
    return None


def extract_last_complete_json(s):
    res = sub_extract_last_complete_json(s)
    if res is None:
        s = s.replace("\{", "{").replace("\}", "}").replace('(', '[').replace(')', ']')
        res = sub_extract_last_complete_json(s)
    if res is None and "\\boxed{" in s:
        boxstart = s.rfind("\\boxed{") + len("\\boxed{")
        boxend = s.rfind("}", boxstart)
        boxcontent = s[boxstart:boxend]
        processed_box_content = boxcontent.replace("\\\\",
                                                   "\\").replace("\\{",
                                                                 "{").replace("\\}",
                                                                              "}").replace('\\left',
                                                                                           '').replace('\\right', '')
        res = sub_extract_last_complete_json(processed_box_content)
    return res

def extract_last_code_block(response_text: str):
    """
    Extracts the content of the last Markdown code block from a string.

    This function searches for all code blocks enclosed in triple backticks (```)
    and returns the content of the very last one it finds. It handles optional
    language specifiers (e.g., ```python).

    Args:
        response_text: The string containing the text to parse, which may
                       include one or more Markdown code blocks.

    Returns:
        The cleaned code content as a string if a code block is found.
        Returns None if no code blocks are present in the input string.
    
    Example:
        >>> text = '''
        ... Here is some text.
        ... ```python
        ... # This is the first code block
        ... print("hello")
        ... ```
        ... And here is the final solution.
        ... ```
        ... # This is the second code block
        ... def my_func():
        ...     return "world"
        ... ```
        ... '''
        >>> extract_last_code_block(text)
        '# This is the second code block\\ndef my_func():\\n    return "world"'
    """
    # The pattern looks for a string starting with ```, optionally followed by a
    # language name and a newline, then captures everything until the next ```.
    # The re.DOTALL flag allows '.' to match newlines, which is crucial for
    # multi-line code blocks.
    pattern = r"```(?:\w*\n)?(.*?)def main_solution(.*?)```"
    
    # Find all non-overlapping matches of the pattern in the string.
    matches = re.findall(pattern, response_text, re.DOTALL)
    
    # If matches were found, return the last one, stripped of leading/trailing whitespace.
    if len(matches) > 0:
        return "def main_solution" + matches[-1][-1].strip()
    
    # If no matches were found, return None.
    return None


def is_close(pred, target, tol=0.001):
    if isinstance(pred, dict) and isinstance(target, dict):
        if pred.keys() != target.keys():
            return False
        return all(is_close(pred[k], target[k], tol) for k in pred)

    elif isinstance(pred, list) and isinstance(target, list):
        if len(pred) != len(target):
            return False
        return all(is_close(p, t, tol) for p, t in zip(pred, target))

    elif isinstance(pred, (int, float)) and isinstance(target, (int, float)):
        try:
            if isinstance(pred, float) or isinstance(target, float):
                # if we have non number, like nan, inf, we should not compare them
                if math.isnan(pred) or math.isnan(target) or math.isinf(pred) or math.isinf(target):
                    return False
                return (abs(pred - target) <= tol * abs(target)) and (int(pred) == int(target))
            return pred == target
        except:
            return False
    else:
        return pred == target


def compute_score_induction(solution, ground_truth):
    from sandbox_fusion import run_code, RunCodeRequest

    ref_input = ground_truth['ref_input']
    ref_code = ground_truth['ref_code']
    functions = ground_truth['functions']
    ref_output = ground_truth['ref_output']

    if not isinstance(ref_input, list):
        ref_input = [ref_input]
    if not isinstance(ref_output, list):
        ref_output = [ref_output]

    all_func_names = re.findall(r"def\s*(.+?)\(", functions)

    main_func = solution[solution.find(f"def {ground_truth['funcname']}"):]
    for function in all_func_names:
        if function + '(' not in main_func:
            print(f"[Error] Function {function} not found in the solution!")
            return 0, 0
        if len(re.findall(r"def\s*" + function, main_func)) >= 1:
            print(f"[Error] The definition of function {function} should not appear in model's response")
            return 0, 0

    for input, output in zip(ref_input, ref_output):
        code = template_check_input.format(solution_prefix=solution_prefix,
                                        refcode=functions + "\n\n" + solution,
                                        funcname=ground_truth['funcname'],
                                        io={
                                            'input': input,
                                            'output': output
                                        },
                                        bypass="False")
        try:
            ret = run_code(RunCodeRequest(code=code, language='python', client_timeout=5))
        except:
            return 0, 0
        if ret.status != "Success":
            print(
                f"[Execution Output]\n  Message: {ret.message}\nSTDOUT: {ret.run_result.stdout}\nSTDERR: {ret.run_result.stderr}"
            )
            print("[Final Score]: 0")
            return 0, 0
    print(
        f"[Execution Output]\n  Message: {ret.message}\nSTDOUT: {ret.run_result.stdout}\nSTDERR: {ret.run_result.stderr}"
    )
    print("[Final Score]: 1")
    return 1, 1


def compute_score_backward(solution, ground_truth):
    from sandbox_fusion import run_code, RunCodeRequest

    ref_input = ground_truth['ref_input']
    ref_code = ground_truth['ref_code']
    ref_output = ground_truth['ref_output']

    if not isinstance(ref_input, list):
        ref_input = [ref_input]
    if not isinstance(ref_output, list):
        ref_output = [ref_output]

    if not isinstance(solution, dict):
        print("The input parameter is not a valid JSON object!")
        return -1, 0

    exact_match = True
    for k, v in ref_input.items():
        if k in solution and solution[k] == v:
            continue
        exact_match = False
    if exact_match:
        return 1, 1

    # Run the code with the input
    code = template_check_input.format(solution_prefix=solution_prefix,
                                       refcode=ref_code,
                                       funcname=ground_truth['funcname'],
                                       io={
                                           'input': solution,
                                           'output': ref_output
                                       },
                                       bypass="False")
    try:
        ret = run_code(RunCodeRequest(code=code, language='python', client_timeout=5))
    except:
        return 0, 0
    if ret.status != "Success":
        print(
            f"[Execution Output]\n  Message: {ret.message}\nSTDOUT: {ret.run_result.stdout}\nSTDERR: {ret.run_result.stderr}"
        )
        print("[Final Score]: 0")
        return 0, 0
    else:
        print(
            f"[Execution Output]\n  Message: {ret.message}\nSTDOUT: {ret.run_result.stdout}\nSTDERR: {ret.run_result.stderr}"
        )
        print("[Final Score]: 1")
        return 1, 1


def compute_score_forward(solution, ground_truth):
    ref_output = ground_truth['ref_output']
    acc = is_close(solution, ref_output)
    if acc:
        print("[Match] Correct!")
        print("[Final Score]: 1")
        return 1, 1
    else:
        print(
            f"[Mismatch] Given the input {json.dumps(ground_truth['ref_input'])}, your predicted output is {json.dumps(solution)}, ground truth is {json.dumps(ref_output)}."
        )
        print("[Final Score]: 0")
        return 0, 0


def compute_score(solution_str, ground_truth, task="codeio-backward"):
    task = task[7:]
    ground_truth = json.loads(ground_truth)

    global do_print
    # do_print = random.randint(1, 256) == 1
    do_print = False

    print("\n" + "=" * 80)
    print(" Processing New Sample ".center(80, '='))
    print(f"[Model Response]\n{solution_str}")
    print(f"[Ground Truth]\nRef Output: {ground_truth['ref_output']}\nRef Input: {ground_truth['ref_input']}")

    if task.startswith('backward') or task.startswith('forward') or (task.startswith('induction') and 'json' in task):
        extraction = extract_last_complete_json
    elif task.startswith('induction'):
        extraction = extract_last_code_block
    try:
        extracted = extraction(solution_str)
        if extracted is None:
            print("Fail to extract a complete and valid json from the model response!")
            return -1, 0
    except Exception as e:
        print(f"Error in extracting JSON: {e}")
        return -1, 0

    print(f"[Extracted]\n{extracted}")

    if (task.startswith('backward') or task.startswith('forward')) and not isinstance(extracted, dict):
        print("The extracted JSON is not a valid JSON object!")
        return -1, 0

    try:
        if task.startswith('backward'):
            if "input" not in extracted:
                print("No field 'input' in the extracted JSON!")
                return -1, 0
            input_param = extracted['input']
            return compute_score_backward(input_param, ground_truth)
        elif task.startswith('forward'):
            if "output" not in extracted:
                print("No field 'output' in the extracted JSON!")
                return -1, 0
            output = extracted['output']
            return compute_score_forward(output, ground_truth)
        elif task.startswith('induction'):
                return compute_score_induction(extracted, ground_truth)
        else:
            raise NotImplementedError(f"Task {task} not implemented.")
    except Exception as e:
        print(f"Error {e}")
        return -1, 0
