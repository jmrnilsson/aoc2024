from functools import reduce
from functools import reduce
from typing import List, Tuple, Any

import numpy as np

from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, print_
from aoc.tools import transpose


_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

def walk_diagonal(s, start_yx: Tuple[int, int], ys, xs):
    word = ""
    for i in range(max(ys, xs)):
        y = i + start_yx[0]
        x = i + start_yx[1]
        if x >= xs:
            break
        if y >= ys:
            break

        word += s[y][x]

    return word

def generate_diagonal(s: List[List[str]] | np.ndarray[Any, np.dtype], ys: int):
    xs = len(s[0])

    for x in range(xs):
        yield walk_diagonal(s, (0, x), ys, xs)

    for y in range(1, ys):
        yield walk_diagonal(s, (y, 0), ys, xs)

def to_word(iterable: List):
    return reduce(lambda r, b: r + b, iterable)

def count(word: str, *patterns: str):
    return sum(word.count(p) for p in patterns)


def solve_1(__input=None):
    """
    :challenge: 18
    :expect: 2575
    """
    input_ = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            input_.append(list(line))


    transposed = transpose(input_)
    yx = len(transposed[0])

    total = 0
    for row in transposed:
        word = to_word(row)
        total += count(word, "XMAS", "SAMX")

    for row in input_:
        word = to_word(row)
        total += count(word, "XMAS", "SAMX")

    words = list(generate_diagonal(input_, yx))

    for row in words:
        word = to_word(row)
        total += count(word, "XMAS", "SAMX")

    rotated = np.rot90(input_)
    words = list(generate_diagonal(rotated, yx))

    for row in words:
        word = to_word(row)
        total += count(word, "XMAS", "SAMX")

    return total

def solve_2(__input=None):
    """
    :challenge: 9
    :expect: 2041
    """
    input_ = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            input_.append(list(line))

    matrix = np.matrix(input_)
    starting_positions = np.argwhere(matrix == "A")

    xmas = {"MAS", "SAM"}
    total = 0
    for starting_position in starting_positions:
        if 0 in starting_position:
            continue

        y, x = starting_position
        if x == matrix.shape[1] - 1 or y == matrix.shape[0] - 1:
            continue

        word_south_east = f"{matrix[y - 1, x - 1]}{matrix[y, x]}{matrix[y + 1, x + 1]}"
        word_north_east = f"{matrix[y + 1, x - 1]}{matrix[y, x]}{matrix[y - 1, x + 1]}"

        if word_south_east in xmas and word_north_east in xmas:
            total += 1

    return total

if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_1, "challenge")
    expect = get_meta_from_fn(solve_1, "expect")
    print_(solve_1, test_input, puzzle_input, challenge, expect)

    challenge = get_meta_from_fn(solve_2, "challenge")
    expect = get_meta_from_fn(solve_2, "expect")
    print_(solve_2, test_input, puzzle_input, challenge, expect)
