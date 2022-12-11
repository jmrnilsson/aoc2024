import re
import re
import sys
from enum import Enum
from typing import List, Tuple, Dict

from defaultlist import defaultlist

from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter

# ICE
_default_puzzle_input = "year_2022/day_01/puzzle.txt"
_default_test_input = "year_2022/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")


challenge_solve_1 = 13140
challenge_solve_2 = """A##..##..##..##..##..##..##..##..##..##..
###...###...###...###...###...###...###.
####....####....####....####....####....
#####.....#####.....#####.....#####.....
######......######......######......####
#######.......#######.......#######....."""


class Day10Operation(Enum):
    ADDX = "addx"
    NOOP = "noop"


def compute(instructions: List[Tuple[Day10Operation, int]], until=260):
    x_s: List[int] = []
    x: int = 1
    i: int = 0
    op: Day10Operation
    number: int
    thread_join: Dict[int, int] = {}
    for n in range(0, until):
        x_s.append(x)
        canonical_x = thread_join.get(n)
        if canonical_x:
            x = canonical_x
            continue

        op, digit = instructions[i] if i < len(instructions) else (Day10Operation.NOOP, -1)
        if op == Day10Operation.ADDX:
            thread_join[n + 1] = x + digit
            i += 1
        elif op == Day10Operation.NOOP:
            i += 1
    return x_s


def solve_1(input_=None):
    """
    test=13140
    expect=13520
    """
    is_test = 1 if "test" in input_ else 0

    lines: List[Tuple[Day10Operation, int]] = []
    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            operation = re.findall(r"^(\w+)?", line)[0]
            number = int(line.replace(operation, "").strip()) if operation == "addx" else -1
            lines.append((Day10Operation(operation), number))

    x_s: List[int] = compute(lines, until=230)
    signal = [v * (n + 1) for n, v in enumerate(x_s)]
    at = [20, 60, 100, 140, 180, 220]
    selected = [(i + 1, v) for i, v in enumerate(signal) if i + 1 in at]

    return sum(v for _, v in selected)


def solve_2(input_=None):
    """
    test=##..##..##..##..##..##..##..##..##..##..
###...###...###...###...###...###...###.
####....####....####....####....####....
#####.....#####.....#####.....#####.....
######......######......######......####
#######.......#######.......#######.....
    expect=###...##..###..#..#.###..####..##..###..
#..#.#..#.#..#.#..#.#..#.#....#..#.#..#.
#..#.#....#..#.####.###..###..#..#.###..
###..#.##.###..#..#.#..#.#....####.#..#.
#....#..#.#....#..#.#..#.#....#..#.#..#.
#.....###.#....#..#.###..####.#..#.###..
    """
    is_test = 1 if "test" in input_ else 0

    lines: List[Tuple[Day10Operation, int]] = []
    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            operation = re.findall(r"^(\w+)?", line)[0]
            number = int(line.replace(operation, "").strip()) if operation == "addx" else -1
            lines.append((Day10Operation(operation), number))

    x_s = compute(lines, until=260)

    acc = defaultlist(list)
    default_sprite = "........................................"
    for i in range(0, len(x_s)):
        x = x_s[i]
        write_out = [x - 1, x, x + 1]
        mod_sprite = list(default_sprite)
        for w in write_out:
            mod_sprite[w] = "#"
        sprite = "".join(mod_sprite)
        pixel_index, row = i % len(sprite), i // len(sprite)
        acc[row].append(sprite[pixel_index])

    return "\n" + "\n".join(["".join(row) for row in acc[:6]])


if __name__ == "__main__":
    """
    PWSH
    while ($true) {python -m aoc.year_2022.day_03.solve -poll; start-sleep -seconds 1} 

    BASH
    watch -n 3 python3 -m aoc.year_2021.day_03.day -p
    """
    poll_printer = PollPrinter(solve_1, solve_2, challenge_solve_1, challenge_solve_2, test_input, puzzle_input)
    poll_printer.run(sys.argv)
