import heapq
import sys
from collections import defaultdict
from typing import List

import numpy as np

from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter

# ICE
_default_puzzle_input = "year_2022/day_01/puzzle.txt"
_default_test_input = "year_2022/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

challenge_solve_1 = "exception"
challenge_solve_2 = "exception"


class Node:
    def __init__(self, position: (), parent: ()):
        self.position = position
        self.parent = parent
        self.g = 0  # Distance to start node
        self.h = 0  # Distance to goal node
        self.f = 0  # Total cost

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    def __repr__(self):
        return '({0},{1})'.format(self.position, self.f)


# def astar_search(_maze: List[List[int]], start, end, weight_min=0):
def astar_search(_maze: np.ndarray, start, end, weight_min=0):
    def is_closer(_open, _neighbor):
        for _node in _open:
            if _neighbor == _node and _neighbor.f >= _node.f:
                return False
        return True

    open_ = []
    closed = []
    start_node = Node(start, None)
    goal_node = Node(end, None)
    node_costs = defaultdict(lambda: float('inf'))
    open_.append(start_node)
    y_len, x_len = len(_maze), len(_maze[0])

    while len(open_) > 0:
        open_.sort()
        current_node = open_.pop(0)
        closed.append(current_node)

        current_node_position = current_node.position
        goal_node_position = goal_node.position
        if current_node_position == goal_node_position:
            path = []
            while current_node != start_node:
                path.append(current_node.position)
                current_node = current_node.parent

            return path[::-1]

        x, y = current_node.position
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        current_weight = _maze[x][y]

        for n in neighbors:
            if 0 > n[0] or y_len <= n[0] or 0 > n[1] or x_len <= n[1]:
                continue

            if not (weight_ := _maze[n[0]][n[1]]):
                continue

            if not (weight_min <= weight_ < current_weight + 2):
                continue

            neighbor = Node(n, current_node)

            if neighbor in closed:
                continue

            cost = 1 + (1 if current_weight != weight_ else 0)

            total_cost = current_node.g + cost
            node_costs[neighbor.position] = min(node_costs[neighbor.position], total_cost)

            # G = combined_cost or combined_cost / count
            neighbor.g = node_costs[neighbor.position]
            neighbor.h = abs(neighbor.position[0] - goal_node.position[0])
            neighbor.h += abs(neighbor.position[1] - goal_node.position[1])
            neighbor.f = neighbor.g + neighbor.h

            if is_closer(open_, neighbor):
                open_.append(neighbor)

    return None


def solve_1(input_=None):
    """
    test=31
    expect=490
    """
    start = None
    end = None

    with open(locate(input_), "r") as fp:
        lines = read_lines(fp)

    maze = np.ones((len(lines), len(lines[0])), dtype=np.int8)

    for y, line in enumerate(lines):
        for x, ch in enumerate(list(line)):
            if ch == "S":
                start = (y, x)
            elif ch == "E":
                end = (y, x)
            else:
                maze[y][x] = ord(ch) - 96

    maze[end[0]][end[1]] = ord("z") - 96 + 1

    path = astar_search(maze, start, end)
    return sum(1 for _ in path)


def solve_2(input_=None):
    """
    test=29
    expect=488
    """
    start, end = None, None

    with open(locate(input_), "r") as fp:
        lines = read_lines(fp)

    maze = np.ones((len(lines), len(lines[0])), dtype=np.int8)

    for y, line in enumerate(lines):
        for x, ch in enumerate(list(line)):
            if ch == "E":
                end = (y, x)
            else:
                maze[y][x] = ord(ch) - 96

    x_len, y_len = len(maze[0]), len(maze)

    matrix = []
    for y in range(y_len):
        x_range = []
        for x in range(x_len):
            x_range.append(maze[y][x])
        matrix.append(x_range)

    maze[end[0]][end[1]] = ord("z") - 96 + 1
    starting_positions_func = np.vectorize(lambda t: t == 1)
    starting_positions = {(x, y) for x, y in np.argwhere(starting_positions_func(maze))}

    shortest = float('inf')
    for start in starting_positions:
        if not (path := astar_search(maze, start, end, weight_min=2)):
            continue
        current = sum(1 for _ in path)
        shortest = min(shortest, current)
        print(f"shortest: {shortest}, current: {current}")

    return shortest


if __name__ == "__main__":
    """
    PWSH
    while ($true) {python -m aoc.year_2022.day_03.solve -poll; start-sleep -seconds 1} 

    BASH
    watch -n 3 python3 -m aoc.year_2021.day_03.day -p
    """
    poll_printer = PollPrinter(solve_1, solve_2, challenge_solve_1, challenge_solve_2, test_input, puzzle_input)
    poll_printer.run(sys.argv)

