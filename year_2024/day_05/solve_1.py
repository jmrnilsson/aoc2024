import sys
from typing import List, Tuple

from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, print_

# import numba


sys.setrecursionlimit(3000)


_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")


def is_ordered(ordering_rules, page_updates):
    ok = []
    for i, page_update in enumerate(page_updates):
        must_see = set()
        for j in range(0, len(page_update)):
            mentioned = set(page_update)
            page_number = page_update[j]
            if page_number in must_see:
                must_see.remove(page_number)

            for left, right in ordering_rules:
                if left == page_number:
                    if left in mentioned and right in mentioned:
                        must_see.add(right)
                    continue
        if len(must_see) == 0:
            ok.append(i)

    return ok


def solve_1(__input=None):
    """
    :challenge: 143
    :expect: 6505
    """
    page_ordering_rules: List[Tuple[int]] = []
    page_updates: List[Tuple[int]] = []
    with open(locate(__input), "r") as fp:
        for ind, line in enumerate(read_lines(fp)):
            if len(line) == 0:
                continue
            if "|" in line:
                page_ordering_rules.append(tuple([int(l) for l in line.split("|")]))
            elif line.index(","):
                page_updates.append(tuple([int(l) for l in line.split(",")]))
            else:
                raise TypeError("Can not parse!")

    pristine_page_updates_indices = is_ordered(page_ordering_rules, page_updates)
    return sum(ppu[len(ppu) // 2] for ppu in (page_updates[i] for i in pristine_page_updates_indices))


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_1, "challenge")
    expect = get_meta_from_fn(solve_1, "expect")
    print_(solve_1, test_input, puzzle_input, challenge, expect)
