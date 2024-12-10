import itertools
import sys
from collections import defaultdict
from typing import List, Tuple, Set, DefaultDict

from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, print_

sys.setrecursionlimit(3000)


_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "sink/test.txt")
test_input_2 = build_location(__file__, "sink/test_2.txt")
test_input_3 = build_location(__file__, "sink/test_3.txt")
test_input_4 = build_location(__file__, "test_4.txt")

def solve(__input=None):
    """
    :challenge: 14
    :expect: 254
    """
    antennas: DefaultDict[Set[Tuple[int, int]]] = defaultdict(set)
    seed: List[Tuple[str]] = []
    with open(locate(__input), "r") as fp:
        for ind, line in enumerate(read_lines(fp)):
            seed.append(list(line))

    width, height = len(seed[0]), len(seed)

    for y, row in enumerate(seed):
        for x, cell in enumerate(row):
            match cell:
                case ".": continue
                case "#": continue
                case _: antennas[cell].add((y, x))

    antinodes: DefaultDict[Set[Tuple[int, int]]] = defaultdict(set)
    for antenna, coords in antennas.items():
        for a, b in itertools.combinations(coords, 2):
            y_displacement, x_displacement = a[0] - b[0], a[1] - b[1]
            left_most, right_most = sorted([a, b], key=lambda entry: entry[1])

            if x_displacement < 0:
                antinode_0 = left_most[0] + y_displacement, left_most[1] + x_displacement
                antinodes[antenna].add(antinode_0)
                antinode_1 = right_most[0] - y_displacement, right_most[1] - x_displacement
                antinodes[antenna].add(antinode_1)
            else:
                antinode_2 = right_most[0] + y_displacement, right_most[1] + x_displacement
                antinodes[antenna].add(antinode_2)
                antinode_3 = left_most[0] - y_displacement, left_most[1] - x_displacement
                antinodes[antenna].add(antinode_3)

    unique_antinodes = {
        (y, x)
        for antenna, coords in antinodes.items()
        for y, x in coords
        if -1 < y < height and -1 < x < width
    }

    return len(unique_antinodes)


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve, "challenge")
    expect = get_meta_from_fn(solve, "expect")
    print_(solve, test_input_4, puzzle_input, challenge, expect)

