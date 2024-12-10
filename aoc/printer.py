import logging
import re
import sys
import traceback
from time import perf_counter
from typing import Callable, List, Literal

from docstring_parser import parse as parse_docs, Docstring

_logger = logger = logging.getLogger(__name__)


def _timing(*args, **kw):
    # https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
    fn, args_ = args[0], args[1:]
    ts = perf_counter()
    result = fn(*args_, **kw)
    te = perf_counter()
    input_file = re.sub("^.*aoc", "", args_[0])
    input_file = re.sub(r"\\", " ", input_file)
    print('%s: %s %s %2.4f ms' % (fn.__name__, input_file, kw, (te - ts) * 1000))
    return result


def put_new_lines():
    """
    Prints two blank lines.
    """
    return print("\n\n")

def put_divider():
    """
    Prints a divider.
    """
    return print(" ---------------------------------")


class ANSIColors:
    HEADER = '\033[95m'
    OK_BLUE = '\033[94m'
    OK_CYAN = '\033[96m'
    OK_GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END_COLOUR = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def try_print(result, color: str, challenge: None | int):
    try:
        result = _print(result, challenge)
    except Exception as e:
        _logger.error(e)
    finally:
        print(f"{color}", result, f"{ANSIColors.END_COLOUR}")

def _print(result, challenge: None | int) -> str | int:
    """
    Uses some heuristics to produce readable outcomes in many scenarios. Ends with capturing stack and prints
    that. Here are some examples:
    - Optimises for printing a simple result but can tack on ancillary, helper information
    - Trims very long outputs so that it does not flood the console and to obliterate previous print
      statements (usually they are more pertinent than an erroneous output).
    - Formats everything as a string in the end.
    - Probably handles ASCII output OK, but not great.
    - Verifies the challenge and tacks on that information if result is well-formed.
    """
    output_collection: List[str] = []
    if bool(isinstance(result, (list, tuple))):
        materialized_output = list(result)
        output_collection.append(str(materialized_output.pop(0)).strip())

        check_challenge = "  ".join(output_collection)
        if re.search(r"^\d+$", check_challenge):
            if str(challenge) == check_challenge:
                output_collection.append("   (OK)")
            else:
                output_collection.append("   (N/OK)")

        for raw_ancillary_info in materialized_output:
            ancillary_info = str(raw_ancillary_info)

            output_collection.append(ancillary_info[:200])
            if len(ancillary_info) > 200:
                output_collection.append("...")

        return "  ".join(output_collection)
    else:
        if isinstance(challenge, int) and isinstance(result, int):
            output = str(result).ljust(20, " ")
            if result == challenge:
                return f"{output}(OK)"
            else:
                return f"{output}(N/OK)"
        return result


def print_(solve_fn: Callable, puzzle_input_example, puzzle_input, challenge: int, answer):
    put_divider()

    try:
        try_print(_timing(solve_fn, puzzle_input_example), ANSIColors.FAIL, challenge)
        try_print(_timing(solve_fn, puzzle_input), ANSIColors.OK_GREEN, answer)
    except Exception as exc:
        traceback.print_exc(file=sys.stdout)

    put_new_lines()

def get_meta_from_fn(fn: Callable, name: Literal["challenge", "expect"]) -> int:
    doc_string: Docstring = parse_docs(fn.__doc__)
    for meta in doc_string.meta:
        if meta.args[0] == name:
            return int(meta.description)

