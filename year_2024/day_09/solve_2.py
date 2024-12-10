import sys
from typing import List

from more_itertools import chunked

from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, print_

sys.setrecursionlimit(30_000)


_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")


class Block:
    """
    Block of disk marshalled by reference.
    """

    def __init__(self, number: int, size: int, free: int = 0):
        self.number = number
        self.size = size
        self.free = free

    def __str__(self):
        return f"Block({self.number}, {self.size}, {self.free})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return self.number


def reshape(_blocks: List[Block]): # Generator[Tuple[int, int, int], None, None]
    blocks = list(_blocks)
    i = len(_blocks) - 1
    while i > 1:
        candidate = _blocks[i]

        if not candidate.size:
            continue

        index_of_candidate = blocks.index(candidate)
        for j in range(0, index_of_candidate, 1):
            block = blocks[j]

            if block.free >= candidate.size:
                blocks.insert(j + 1, Block(candidate.number, candidate.size, block.free - candidate.size))
                candidate.free = candidate.size + candidate.free
                candidate.size = 0
                block.free = 0
                break

        i -= 1

    return blocks


def solve(__input=None):
    """
    :challenge: 2858
    :expect: 6307653242596
    """
    with open(locate(__input), "r") as fp:
        lines = [int(i) for i in list(read_lines(fp)[0])]

    blocks: List[Block] = [
        Block(number, *chunk) for number, chunk in enumerate(chunked(lines, 2))
    ]

    index = 0
    total = 0
    for v in reshape(blocks):
        total += sum(i * v.number for i in range(index, index + v.size))
        index += v.size + v.free

    return total


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve, "challenge")
    expect = get_meta_from_fn(solve, "expect")
    print_(solve, test_input, puzzle_input, challenge, expect)
