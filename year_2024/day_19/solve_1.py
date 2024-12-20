import itertools
import operator
import re
import statistics
import sys
from collections import Counter, OrderedDict, defaultdict
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Callable, Tuple, Literal, Set, Generator, Any

import more_itertools
# import numba
import numpy as np
from defaultlist import defaultlist
from more_itertools import windowed, chunked
from more_itertools.recipes import sliding_window
from automata.fa.nfa import NFA

from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter
from aoc.printer import get_meta_from_fn, print_, ANSIColors, print2
from aoc.tests.test_fixtures import get_challenges_from_meta
from aoc.tools import transpose
from year_2021.day_05 import direction

# from numba import cuda, np as npc, vectorize

# requires https://developer.nvidia.com/cuda-downloads CUDA toolkit. There are some sketchy versions in pip, but it's
# almost impossible to find the right versions.
# - pip install nvidia-pyindex
# - pip install nvidia-cuda-runtime-cuXX
# python -m numba -s
# print(numba.__version__)
# print(cuda.gpus)
# print(cuda.detect())

# if cuda.is_available():
#     print("CUDA is available!")
#     print("Device:", cuda.get_current_device().name)
# else:
#     print("CUDA is not available.")

sys.setrecursionlimit(30_000)

_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")
test_input_3 = build_location(__file__, "test_3.txt")
test_input_4 = build_location(__file__, "test_4.txt")
test_input_5 = build_location(__file__, "test_5.txt")


class TowelAutomaton:

    def __init__(self, towels):
        self.towels = towels

    def accepts(self, word):
        last_states = {word}
        while 1:
            canonical_state = set(last_states)
            for towel in self.towels:
                current_state = set()
                for state in last_states:
                    if state == "":
                        return True

                    current_state.add(state)
                    if state.startswith(towel):
                        current_state.add(state[len(towel):])

                last_states = current_state

            if canonical_state == last_states:
                break

        return False


def solve_(__input=None):
    """
    :challenge: 6
    :expect: 374
    """
    words = []
    with open(locate(__input), "r") as fp:
        lines = read_lines(fp)
        towels = lines[0].split(", ")
        for word in lines[1:]:
            words.append(word)

    nfa = TowelAutomaton(towels)

    n = 0
    total = list()
    for word in words:
        if nfa.accepts(word):
            total.append(word)
            n += 1

    return sum(1 for _ in total)


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
