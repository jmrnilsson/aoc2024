import heapq
import itertools
import sys
from enum import Enum
from typing import List, Tuple, Generator, Self

import numpy as np

from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, ANSIColors, print2

sys.setrecursionlimit(30_000)

_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")
test_input_3 = build_location(__file__, "test_3.txt")
test_input_4 = build_location(__file__, "test_4.txt")
test_input_5 = build_location(__file__, "test_5.txt")


class Heading(Enum):
    N = 0
    E = 1
    S = 2
    W = 3

    def von_neumann(self) -> Tuple[int, int]:
        match self:
            case Heading.N: return -1, 0
            case Heading.S: return 1, 0
            case Heading.W: return 0, -1
            case Heading.E: return 0, 1

    def make_turns(self) -> List[Self]:
        if self == Heading.N or self == Heading.S:
            return [Heading.W.value, Heading.E.value]
        return [Heading.N.value, Heading.S.value]

def neighbors_and_cost_delta(heading: int, y: int, x: int, grid) -> Generator[Tuple[Heading, int, int, int], None, None]:
    heading_explorer = Heading(heading)
    for turned in heading_explorer.make_turns():
        yield turned, y, x, 1000

    dy, dx = heading_explorer.von_neumann() # a move
    new_node = y + dy, x + dx

    cell_value = str(grid[new_node])
    if cell_value != "#":
        yield heading, new_node[0], new_node[1], 1


def heuristics_manhattan(a: Tuple[int, int], b: Tuple[int, int]):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def dijkstra(begin_pos: Tuple[int, int], begin_heading: Heading, end: Tuple[int, int], costs, grid):
    """
    Dijkstra-variant with PQ and heuristics for cut-off and priority. Not quite A* either.
    """
    begin = (begin_heading.value, *begin_pos)
    pq: List[Tuple[int, int, int, int]] = [(0, *begin)]
    costs[begin_heading.value, begin[1], begin[2]] = 0
    seen = set()
    while pq:
        _, heading, y, x, = heapq.heappop(pq)
        current = heading, y, x
        current_cost = costs[heading, y, x]

        if current in seen:
            continue

        leads = list(neighbors_and_cost_delta(heading, y, x, grid))
        for neighbor_heading, ny, nx, delta_cost in leads:
            previous_cost = costs[neighbor_heading, ny, nx]
            total_cost = current_cost + delta_cost
            neighbor = neighbor_heading, ny, nx
            if previous_cost > total_cost:
                costs[neighbor] = total_cost
                if neighbor in seen:
                    seen.remove(neighbor)

            h = heuristics_manhattan((ny, nx), end)
            heapq.heappush(pq, (h, neighbor_heading, ny, nx))

        seen.add(current)



def solve_(__input=None):
    """
    :challenge: 7036
    :challenge_2: 11048
    :expect: 73432
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            lines.append(list(line))

    grid = np.matrix(lines, dtype=str)
    cost_shape = (4, *grid.shape)
    costs = np.full(cost_shape, dtype=np.int64, fill_value=sys.maxsize)
    begin, = [(int(y), int(x)) for y, x in np.argwhere(grid == 'S')]
    end, = [(int(y), int(x)) for y, x in np.argwhere(grid == 'E')]
    begin_heading = Heading.E

    dijkstra(begin, begin_heading, end, costs, grid)

    # For troubleshooting
    cost_summary = np.zeros(grid.shape)

    for z, y, x in itertools.product(range(cost_shape[0]), range(cost_shape[1]), range(cost_shape[2])):
        if (cost_summary[y, x] == 0 or costs[z, y, x] < cost_summary[y, x]) and costs[z, y, x] != sys.maxsize:
            cost_summary[y, x] = costs[z, y, x]

    return costs[:,end[0],end[1]].min()


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    challenge_2 = get_meta_from_fn(solve_, "challenge_2")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, test_input_2, challenge_2, ANSIColors.WARNING)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
