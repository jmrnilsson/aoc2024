import sys
from typing import List, Tuple, Generator

from more_itertools import chunked

from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, print_

sys.setrecursionlimit(30_000)


_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")


def reshape(blocks: List[Tuple[int, int, int]]) -> Generator[Tuple[int, int], None, None]:
    if len(blocks) < 1:
        return

    number_head, size_head, free_head = blocks[0]
    if size_head:
        yield number_head, size_head

    if free_head < 1 < len(blocks):
        for res in reshape(blocks[1:]):
            yield res

    elif free_head and len(blocks) > 1:
        number_tail, size_tail, free_tail = blocks[-1]

        if size_tail == free_head:
            yield number_tail, size_tail
            for res in reshape(blocks[1:-1]):
                yield res

        if free_head > size_tail:
            yield number_tail, size_tail
            new_blocks = [(0, 0, free_head - size_tail)] + blocks[1:-1]
            for res in reshape(new_blocks):
                yield res

        if size_tail > free_head:
            yield number_tail, free_head
            new_blocks = blocks[1:-1] + [(number_tail, size_tail - free_head, free_tail)]

            for res in reshape(new_blocks):
                yield res


def solve(__input=None):
    """
    :challenge: 1928
    :expect: 6283170117911
    """
    with open(locate(__input), "r") as fp:
        lines = [int(i) for i in list(read_lines(fp)[0])]

    to_block = lambda chunk_: (chunk_[0], chunk_[1] if len(chunk_) > 1 else 0)
    blocks = [(number, *to_block(chunk)) for number, chunk in enumerate(chunked(lines, 2))]
    defragmented = reshape(blocks)

    j = 0
    total = 0
    for number, size in defragmented:
        total += sum(i * number for i in range(j, j + size))
        j += size

    return total


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve, "challenge")
    expect = get_meta_from_fn(solve, "expect")
    print_(solve, test_input, puzzle_input, challenge, expect)

