import json
import operator
import re
import sys
from collections import Counter, OrderedDict, defaultdict

import more_itertools
import numpy as np
from more_itertools import sliding_window

from aoc import tools
from aoc.helpers import timing, locate, Printer, build_location, test_nocolor, puzzle_nocolor, read_lines
from aoc.poll_printer import PollPrinter

# ICE
_default_puzzle_input = "year_2022/day_01/puzzle.txt"
_default_test_input = "year_2022/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

challenge_solve_1 = 5
challenge_solve_2 = "exception"


def solve_1(input_=None):
    """
    test=7
    expect=1198
    """
    start = 4 if "test" in input_ else 9

    with open(locate(input_), "r") as fp:
        lines = read_lines(fp)[:start]

    n = 4
    for i, w in enumerate(sliding_window(list(lines[0]), n)):
        if len(set(w)) == n:
            return i + n


def solve_2(input_=None):
    """
    test=19
    expect=3120
    """
    start = 4 if "test" in input_ else 9

    with open(locate(input_), "r") as fp:
        lines = read_lines(fp)[:start]

    n = 14
    for i, w in enumerate(sliding_window(list(lines[0]), n)):
        if len(set(w)) == n:
            return i + n


if __name__ == "__main__":
    """
    PWSH
    while ($true) {python -m aoc.year_2022.day_03.solve -poll; start-sleep -seconds 1} 

    BASH
    watch -n 3 python3 -m aoc.year_2021.day_03.day -p
    """
    poll_printer = PollPrinter(solve_1, solve_2, challenge_solve_1, challenge_solve_2, test_input, puzzle_input)
    args = sys.argv[1:]
    if not args:
        poll_printer.print_timed()
    elif re.match("^-poll$|^-p$", args[0]):
        poll_printer.poll_print()
    elif re.match("^-json1$|^-j1$", args[0]):
        poll_printer.poll_json_1()
    elif re.match("^-json2$|^-j2$", args[0]):
        poll_printer.poll_json_2()


