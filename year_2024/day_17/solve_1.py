import re
import sys

from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, ANSIColors, print2
from year_2024.day_17.chronospatial_computer import ChronospatialComputer


sys.setrecursionlimit(30_000)

_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")
test_input_3 = build_location(__file__, "test_3.txt")
test_input_4 = build_location(__file__, "test_4.txt")
test_input_5 = build_location(__file__, "test_5.txt")


def solve_(__input=None):
    # challenge 4,6,3,5,6,3,5,2,1,0.
    # expect 2,1,3,0,5,2,3,7,1
    """
    :challenge_2: -1
    :expect: -1
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            lines.append(line)

    a, b, c = [int(m) for m in re.findall(r"\d+", "".join(lines[:3]))]
    optcodes = [int(m) for m in re.findall(r"\d+", lines[3])]

    outcome = []
    computer = ChronospatialComputer(a, b, c, optcodes, outcome.append)

    while optcode := computer.next():
        computer.set_operation(optcode[0])
        computer.set_operand(optcode[1])
        computer.apply()

    return ",".join(str(o) for o in outcome)

if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
