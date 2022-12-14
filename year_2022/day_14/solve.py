import abc
import copy
import sys
from typing import Tuple, List, Callable

import numpy as np

from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter

# ICE
_default_puzzle_input = "year_2022/day_01/puzzle.txt"
_default_test_input = "year_2022/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

challenge_solve_1 = 24
challenge_solve_2 = "exception"


class SandPhysics(object):

    def __init__(self, matrix):
        self.matrix = matrix

    @abc.abstractmethod
    def down(self, sand):
        pass

    @abc.abstractmethod
    def left_or_right(self, sand):
        pass

    def pour(self, sand):
        previous = None
        while previous != sand:
            previous = sand
            sand = self.down(sand)
            sand = self.left_or_right(sand)
        return sand


class SandPhysicsPart1(SandPhysics):
    def left_or_right(self, sand):
        x, y = sand
        if self.matrix[y + 1, x - 1] == 0:
            y += 1
            x -= 1
        elif self.matrix[y + 1, x + 1] == 0:
            y += 1
            x += 1
        return x, y

    def down(self, sand):
        x, y = sand
        pristine = True
        while self.matrix[y, x] == 0:
            y += 1
            pristine = False
        return x, (y - 1 if not pristine else y)


class SandPhysicsPart2(SandPhysics):

    def left_or_right(self, sand):
        x, y = sand
        if len(self.matrix) > y + 1:
            if self.matrix[y + 1, x - 1] == 0:
                y += 1
                x -= 1
            elif self.matrix[y + 1, x + 1] == 0:
                y += 1
                x += 1
            return x, y
        return sand

    def down(self, sand):
        x, y = sand
        pristine = True
        while len(self.matrix) > y + 1 and self.matrix[y, x] == 0:
            y += 1
            pristine = False
        return x, (y - 1 if not pristine else y)


def parse_line(line):
    previous = None
    for split_line in line.split(" -> "):
        x, y = [int(d) for d in split_line.split(",")]
        if previous:
            if previous[0] == x:
                first, second = sorted([previous[1], y])
                for y_ in range(first, second + 1):
                    yield x, y_
            if previous[1] == y:
                first, second = sorted([previous[0], x])
                for x_ in range(first, second + 1):
                    yield x_, y
        previous = x, y


def pour_sand(seed: List[Tuple[int, int]], sand_physics_factory: Callable, expand_x: int = 0, expand_y: int = 0):
    max_x, max_y = max(x for x, _ in seed),  max(y for _, y in seed)
    matrix = np.zeros((max_y + 1 + expand_y, max_x + 1 + expand_x))

    for x, y in seed:
        matrix[y, x] = 1

    n, sand_physics = 0, sand_physics_factory(matrix)
    while 1:
        current_matrix = copy.deepcopy(matrix)

        try:
            x, y = sand_physics.pour((500, 0))
            matrix[y, x] = 2
        except IndexError:
            break

        n += 1

        if np.sum(matrix) == np.sum(current_matrix):
            break

    is_sand = np.vectorize(lambda t: t == 2)
    return sum(1 for _ in np.argwhere(is_sand(matrix)))


def solve_1(input_=None):
    """
    test=24
    expect=897
    """
    seed = []

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            seed += list(parse_line(line))

    return pour_sand(seed, lambda mat: SandPhysicsPart1(mat))


def solve_2(input_=None):
    """
    test=93
    expect=26683
    """
    seed = []

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            seed += list(parse_line(line))

    max_x, max_y = max(x for x, _ in seed), max(y for _, y in seed)
    matrix = np.zeros((max_y + 2, max_x + 1000))

    for x, y in seed:
        matrix[y, x] = 1

    return pour_sand(seed, lambda mat: SandPhysicsPart2(mat), expand_y=1, expand_x=200)


if __name__ == "__main__":
    """
    PWSH
    while ($true) {python -m aoc.year_2022.day_03.solve -poll; start-sleep -seconds 1} 

    BASH
    watch -n 3 python3 -m aoc.year_2021.day_03.day -p
    """
    poll_printer = PollPrinter(solve_1, solve_2, challenge_solve_1, challenge_solve_2, test_input, puzzle_input)
    poll_printer.run(sys.argv)

