import json
import operator
import re
import sys
from collections import Counter, OrderedDict, defaultdict

import numpy as np

from aoc import tools
from aoc.helpers import timing, locate, Printer, build_location, test_nocolor, puzzle_nocolor, read_lines
from aoc.poll_printer import PollPrinter

# ICE
_default_puzzle_input = "year_2022/day_01/puzzle.txt"
_default_test_input = "year_2022/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

challenge_solve_1 = "exception"
challenge_solve_2 = "exception"


def solve_1(input_=None):
    """
    test=58
    expect=1598415
    """
    start = 1 if "test" in input_ else 0

    seed = []
    seed_set = set()
    seed_dict = {}
    seed_counter = Counter()
    seed_ordered_dict = OrderedDict()
    seed_np = np.array([])
    seed_np_2d = np.array([[]])
    seed_default_dict = defaultdict(list)

    area = 0

    with open(locate(input_), "r") as fp:
        # lines = [li.strip() for li in fp.readlines()]
        for line in fp.readlines():
            if line == "\n":
                continue
            l, w, h = [int(i) for i in re.findall(r"\d+", line)]
            area += 2*l*w + 2*w*h + 2*h*l
            area += operator.mul(*sorted([l, w, h])[:-1])

    return area


def solve_2(input_=None):
    """
    test=34
    expect=3812909
    """
    start = 1 if "test" in input_ else 0

    total = 0
    area = 0

    with open(locate(input_), "r") as fp:
        lines = read_lines(fp)[:start]

    matrix = tools.to_matrix(lines, " ", replace_chars="[]")
    tools.bitmap_print_matrix(matrix)
    print(tools.get_vectors(matrix, str_join=True))

    for line in lines:
        if line == "\n":
            continue
        l, w, h = [int(i) for i in re.findall(r"\d+", line)]
        area += operator.add(*sorted([l, w, h])[:-1])*2
        area += l * w * h

    return total


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


