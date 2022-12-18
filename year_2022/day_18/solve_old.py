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


def solve_2(input_=None):
    """
    test=58
    expect=1598415
    """
    is_test = 1 if "test" in input_ else 0

    pixels = set()
    neighbors = set()
    surface_area = 0
    void = 0

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            pixel = tuple(int(i) for i in re.findall(r"\d+", line))
            pixels.add(pixel)

    trapping_surfaces = set()
    for p in pixels:
        xys = {(p[0], p[1]) for p in pixels}
        yzs = {(p[1], p[2]) for p in pixels}
        xzs = {(p[0], p[2]) for p in pixels}

        for x, y in xys:
            min_z = min(p[2] for p in pixels if x == p[0] and y == p[1]) + 1
            max_z = max(p[2] for p in pixels if x == p[0] and y == p[1]) - 1

            for z in range(min_z, max_z + 1):
                candidate = (x, y, z)
                if candidate not in pixels:
                    trapping_surfaces.add((x, y, z))

        for y, z in yzs:
            min_x = min(p[0] for p in pixels if y == p[1] and z == p[2]) + 1
            max_x = max(p[0] for p in pixels if y == p[1] and z == p[2]) - 1

            for x in range(min_x, max_x + 1):
                if (x, y, z) not in pixels:
                    trapping_surfaces.add((x, y, z))

        for x, z in xzs:
            min_y = min(p[1] for p in pixels if z == p[2] and x == p[0]) + 1
            max_y = max(p[1] for p in pixels if z == p[2] and x == p[0]) - 1

            for y in range(min_y, max_y + 1):
                if (x, y, z) not in pixels:
                    trapping_surfaces.add((x, y, z))

    trapped_air = set()
    for trapping_surface in trapping_surfaces:
        neighbors = set(von_neumann_neighbors_3d(trapping_surface))
        diff = trapping_surfaces.difference(neighbors)
        if not diff:
            trapped_air.add(trapping_surface)

    for p in pixels:
        for neighbor in von_neumann_neighbors_3d(p):
            if neighbor not in pixels and neighbor not in trapped_air:
                surface_area += 1

    # return sum(1 for _ in droplet_inner_bounds)
    return surface_area - void



def von_neumann_neighbors_3d(p):
    return [
        (p[0] - 1, p[1], p[2]),
        (p[0] + 1, p[1], p[2]),
        (p[0], p[1] - 1, p[2]),
        (p[0], p[1] + 1, p[2]),
        (p[0], p[1], p[2] - 1),
        (p[0], p[1], p[2] + 1),
    ]


# def vectorize_points(points):
#     min_x, min_y, min_z = min(points, key=lambda p: p[0])[0], min(points, key=lambda p: p[1])[1], min(points, key=lambda p: p[2])[2]
#     max_x, max_y, max_z = max(points, key=lambda p: p[0])[0], max(points, key=lambda p: p[1])[1], max(points, key=lambda p: p[2])[2]
#     vectorized = np.zeros((max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1), dtype=np.int8)
#     for p in points:
#         vectorized[p[0] - min_x, p[1] - min_y, p[2] - min_z] = 1
#     return vectorized

def _old_solve_2(input_=None):
    """
    test=58
    expect=1598415
    """
    is_test = 1 if "test" in input_ else 0

    pixels = set()
    droplets = set()

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            pixel = tuple(int(i) for i in re.findall(r"\d+", line))
            pixels.add(pixel)

    for p in pixels:
        for n in von_neumann_neighbors_3d(p):
            neighbors_2d = von_neumann_neighbors_3d(n)
            if all([i in pixels for i in neighbors_2d]) and n not in pixels:
                droplets.add(n)

    _sorted_droplets = sorted(droplets, key=lambda p: p)

    n = 0
    seen_neighbors = set()
    for start in droplets:
        residual = {start}
        seen = set()
        while residual:
            node = residual.pop()
            n += 1
            if n % 1000 == 0:
                print(f"queue size: {len(residual)}")
            seen.add(node)
            if node not in droplets:
                seen_neighbors.add(node)
            for neighbor in von_neumann_neighbors_3d(node):
                # seen_neighbors.add(neighbor)
                if neighbor not in seen and neighbor in pixels:
                    # if node in seen_neighbors:
                    #    seen_neighbors.remove(node)
                    residual.add(neighbor)

    # a = vectorize_points(seen_neighbors)

    neighbors_set = set()
    neighbors_n = 0
    for p in seen_neighbors:
        for neighbor in von_neumann_neighbors_3d(p):
            if neighbor not in pixels and neighbor not in droplets and neighbor not in seen_neighbors:
                neighbors_n += 1
                neighbors_set.add(neighbor)

    # test 58 - 3134 too high
    # test - 1547
    # return sum(1 for _ in neighbors_set)
    return neighbors_n


if __name__ == "__main__":
    """
    PWSH
    while ($true) {python -m aoc.year_2022.day_03.solve -poll; start-sleep -seconds 1} 

    BASH
    watch -n 3 python3 -m aoc.year_2021.day_03.day -p
    """
    poll_printer = PollPrinter(solve_1, solve_2, challenge_solve_1, challenge_solve_2, test_input, puzzle_input)
    poll_printer.run(sys.argv)

