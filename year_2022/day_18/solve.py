import json
import operator
import re
import sys
from collections import Counter, OrderedDict, defaultdict
import heapq as heap
import numpy as np

from aoc import tools
from aoc.helpers import timing, locate, Printer, build_location, test_nocolor, puzzle_nocolor, read_lines
from aoc.poll_printer import PollPrinter

# ICE
_default_puzzle_input = "year_2022/day_01/puzzle.txt"
_default_test_input = "year_2022/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

challenge_solve_1 = 64
challenge_solve_2 = 58


def solve_1(input_=None):
    """
    test=58
    expect=1598415
    """
    is_test = 1 if "test" in input_ else 0

    pixels = set()
    neighbors = set()
    neighbors_n = 0

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            pixel = tuple(int(i) for i in re.findall(r"\d+", line))
            pixels.add(pixel)

    for p in pixels:
        for neighbor in von_neumann_neighbors_3d(p):
            if neighbor not in pixels:
                neighbors_n += 1

    return neighbors_n


def find_a_start_in_void(pixels, droplet_part):
    for pixel in pixels:
        for neighbor in von_neumann_neighbors_3d(pixel):
            try:
                if neighbor not in pixels:
                    x, y, z = neighbor
                    min_z = min(p[2] for p in pixels if x == p[0] and y == p[1])
                    max_z = max(p[2] for p in pixels if x == p[0] and y == p[1])
                    if min_z < neighbor[2] < max_z:
                        min_x = min(p[0] for p in pixels if y == p[1] and z == p[2])
                        max_x = max(p[0] for p in pixels if y == p[1] and z == p[2])
                        if min_x < neighbor[0] < max_x:
                            min_y = min(p[1] for p in pixels if z == p[2] and x == p[0])
                            max_y = max(p[1] for p in pixels if z == p[2] and x == p[0])
                            if min_y < neighbor[1] < max_y:
                                if neighbor not in droplet_part:
                                    return neighbor
            except ValueError:
                continue


def solve_2(input_=None):
    """
    test=58
    expect=1598415
    """
    is_test = 1 if "test" in input_ else 0

    pixels = set()
    droplet = set()
    surface_area = 0
    void_surface_area = 0
    void = 0

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            pixel = tuple(int(i) for i in re.findall(r"\d+", line))
            pixels.add(pixel)

    for p in pixels:
        for neighbor in von_neumann_neighbors_3d(p):
            if neighbor not in pixels:
                surface_area += 1

    # simplified dijkstra
    droplet = set()
    n = 0
    while start := find_a_start_in_void(pixels, droplet):

        residual = {start}
        seen = set()
        while residual:
            node = residual.pop()
            n += 1
            if n % 1000 == 0:
                print(f"queue size: {len(residual)}")
            seen.add(node)
            droplet.add(node)
            for neighbor in von_neumann_neighbors_3d(node):
                if neighbor not in seen and neighbor not in pixels:
                    residual.add(neighbor)

    for p in droplet:
        for neighbor in von_neumann_neighbors_3d(p):
            if neighbor not in droplet and neighbor:
                void_surface_area += 1

    # 3134 too high
    # 2226 too high
    return surface_area - void_surface_area


def von_neumann_neighbors_3d(p):
    return [
        (p[0] - 1, p[1], p[2]),
        (p[0] + 1, p[1], p[2]),
        (p[0], p[1] - 1, p[2]),
        (p[0], p[1] + 1, p[2]),
        (p[0], p[1], p[2] - 1),
        (p[0], p[1], p[2] + 1),
    ]

if __name__ == "__main__":
    """
    PWSH
    while ($true) {python -m aoc.year_2022.day_03.solve -poll; start-sleep -seconds 1} 

    BASH
    watch -n 3 python3 -m aoc.year_2021.day_03.day -p
    """
    poll_printer = PollPrinter(solve_1, solve_2, challenge_solve_1, challenge_solve_2, test_input, puzzle_input)
    poll_printer.run(sys.argv)

