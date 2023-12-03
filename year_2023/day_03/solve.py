import itertools
import operator
import sys
from collections import defaultdict
from re import Match, finditer as re_finditer, findall as re_findall
from typing import List, Tuple, Set

from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter

# Override only if necessary
_default_puzzle_input = "year_2023/day_01/puzzle.txt"
_default_test_input = "year_2023/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

challenge_1 = 4361
challenge_2 = 467835


def heuristic_distance_from(number_vector: Tuple[int, int, int, int], symbols: Set[Tuple[int, int]]):
    y, xs, xe, number = number_vector
    for x, (symbol_y, symbol_x) in itertools.product(range(xs, xe), symbols):
        if 2 > abs(y - symbol_y) and 2 > abs(x - symbol_x):
            return True
    return False

def solve_1(input_=None):
    """
    :challenge: 4361
    :expect: 535235
    """
    with open(locate(input_), "r") as fp:
        lines = read_lines(fp)

    y_matches: List[List[Tuple[Match, str]]] = [
        [
            (match, pattern)
            for pattern in ("\d+", "\S")
            for match in re_finditer(pattern, line)
            if match.group() != "." and (pattern != "\S" or not re_findall("\d", match.group()))
        ]
        for line in lines
    ]

    symbols: Set[Tuple[int, int]] = set()
    numbers: Set[Tuple[int, int, int, int]] = set()
    for y, y_match in enumerate(y_matches):
        for match, pattern in y_match:
            xs, xe, mg = match.start(), match.end(), match.group()
            if pattern == '\\d+':
                # y, x_start, x_end, number
                numbers.add((y, xs, xe, int(mg)))
            else:
                # y, x
                symbols.add((y, xs))

    return sum(n[-1] for n in numbers if heuristic_distance_from(n, symbols))


def solve_2(input_=None):
    """
    :challenge: 467835
    :expect: 79844424
    """
    with open(locate(input_), "r") as fp:
        lines = read_lines(fp)

    y_matches: List[List[Tuple[Match, str]]] = [
        [
            (match, pattern)
            for pattern in ("\d+", "\*")
            for match in re_finditer(pattern, line)
        ]
        for line in lines
    ]

    symbols: Set[Tuple[int, int]] = set()
    numbers: Set[Tuple[int, int, int, int]] = set()
    for y, y_match in enumerate(y_matches):
        for match, pattern in y_match:
            if pattern == '\\d+':
                # y, x_start, x_end, number
                numbers.add((y, match.start(), match.end(), int(match.group())))
            else:
                symbols.add((y, match.start()))

    gears = defaultdict(set)
    for x_vectors in numbers:
        y, xs, xe, number = x_vectors
        for x, symbol in itertools.product(range(xs, xe), symbols):
            sy, sx = symbol
            if 2 > abs(y - sy) and 2 > abs(x - sx):
                gears[symbol].add(x_vectors)

    return sum(operator.mul(*map(lambda n: n[-1], numbers)) for numbers in gears.values() if len(numbers) == 2)


if __name__ == "__main__":
    poll_printer = PollPrinter(solve_1, solve_2, challenge_1, challenge_2, test_input, puzzle_input)
    poll_printer.run(sys.argv)

