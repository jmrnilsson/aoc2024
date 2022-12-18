import json
import operator
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from itertools import cycle
from typing import List, Tuple, Callable

import numpy as np

from bitarray import bitarray
from bitarray.util import ba2int

from aoc import tools
from aoc.helpers import timing, locate, Printer, build_location, test_nocolor, puzzle_nocolor, read_lines
from aoc.poll_printer import PollPrinter

# ICE
_default_puzzle_input = "year_2022/day_01/puzzle.txt"
_default_test_input = "year_2022/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

challenge_solve_1 = "exception"
challenge_solve_2 = "exception"


def apply(coords: List[Tuple[int, int]], func: Callable):
    new_coords = []
    for coord in coords:
        new_coords.append(func(coord))
    return new_coords


def center(p, name):
    if name in ("horizontal_line", "reverse_l", "plus", "bloc"):
        return p[0] + 2, p[1]
    elif name in ("vertical_line"):
        return p[0] + 2, p[1]
    else:
        raise ValueError(f"Unknown name: {name}")


# simulate a rock falling down a cavern and bouncing off the walls and may be stopped by previous rocks that have fallen
# to the ground or on top of other rocks. Can't move right or left if blocked by a rock. Can only move down if not
# blocked by another rock. If blocked by a rock on the bottom it will stop, and another rock with different shape will
# fall.
def fall(movements, n=2023, is_test=True):
    horizontal_line = [(0, 0), (1, 0), (2, 0), (3, 0)]
    plus = [(0, 1), (1, 1), (2, 1), (1, 0), (1, 2)]
    reverse_l = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    vertical_line = [(0, 0), (0, 1), (0, 2), (0, 3)]
    bloc = [(0, 0), (1, 0), (0, 1), (1, 1)]

    # rocks = [np.asarray(a) for a in [horizontal_line, plus, reverse_l, vertical_line, bloc]]
    rocks_ = [horizontal_line, plus, reverse_l, vertical_line, bloc]
    rock_names = ["horizontal_line", "plus", "reverse_l", "vertical_line", "bloc"]
    rocks = [(n, rock)for n, rock in zip(rock_names, rocks_)]
    walls = {-1, 7}
    m = 0
    max_y = 0
    shape = set()
    down = lambda p: (p[0], p[1] - 1)

    wind = 0
    wind_mod = len(movements)
    for name, rock in cycle(rocks):
        rock = apply(rock, lambda p: center(p, name))
        out_of_bounds = 3 + max_y
        # + (1 if name in ["horizontal_line", "plus", "reverse_l"] else 0)
        up = lambda p: (p[0], p[1] + out_of_bounds)
        rock = apply(rock, up)
        y_min = min([p[1] for p in rock])
        assert y_min - max_y == 3
        last_rock = rock
        breaker = False
        while not breaker:
            movement = movements[wind % wind_mod]
            # for movement in movements:
            if movement == ">":
                sideways = lambda p: (p[0] + 1, p[1])
            else:
                sideways = lambda p: (p[0] - 1, p[1])
            before_wall = rock
            rock = apply(rock, sideways)
            for x, y in rock:
                if x in walls:
                    rock = before_wall
                elif (x, y) in shape:
                    rock = before_wall
            before_down = rock
            rock = apply(rock, down)
            for x, y in rock:
                if y == -1:
                    rock = before_down
                    breaker = True
                    # wind -= 1
                    # if wind < 0:
                    #     wind = len(movements) - 1
                    break
                elif (x, y) in shape:
                    rock = before_down
                    breaker = True
                    # wind -= 1
                    # if wind < 0:
                    #     wind = len(movements) - 1
                    break
            # if not breaker:
            wind += 1
            # else:
            #     wind -= 1
            #     if wind < 0:
            #         wind = len(movements) - 1

            last_rock = rock

        for coords in rock:
            if coords[1] in shape:
                raise ValueError(f"Duplicate y: {coords}")
            shape.add(coords)

        if is_test:
            if m == 0:
                assert rock == [(2, 0), (3, 0), (4, 0), (5, 0)]
            if m == 1:
                assert rock == [(2, 2), (3, 2), (4, 2), (3, 1), (3, 3)]
            if m == 2:
                assert rock == [(0, 3), (1, 3), (2, 3), (2, 4), (2, 5)]
            if m == 3:
                assert rock == [(4, 3), (4, 4), (4, 5), (4, 6)]
            if m == 4:
                assert rock == [(4, 7), (5, 7), (4, 8), (5, 8)]
            if m == 5:
                assert rock == [(1, 9), (2, 9), (3, 9), (4, 9)]

        max_y = max([y for _, y in shape]) + 1
        m += 1

        # _max_x = max([p[0] for p in shape])
        # display = np.zeros((max_y, _max_x + 2))
        #
        # for x, y in shape:
        #     display[y, x] = 1
        #
        # if 5 <= m < 100:
        #     d = np.rot90(display, 1)
        #     d = np.rot90(d, 1)
        #     d = np.fliplr(d)
        #     ys = [y for y in range(d.shape[0]) if np.any(d[y, :])]
        #     for y in ys:
        #         print("".join(["#" if d[y, x] else "." for x in range(d.shape[1])]))
        #
        #     print("\n------------------\n")

        if m == n - 1:
            break

    return abs(max_y)


def get_bytes(bits):
    done = False
    while not done:
        byte = 0
        for _ in range(0, 6):
            try:
                bit = next(bits)
            except StopIteration:
                bit = 0
                done = True
            byte = (byte << 1) | bit
        yield byte


def fall_p2(movements, n=2023, is_test=True):
    has_found_offset = False

    last_seen = dict()
    iters = defaultdict(set)

    horizontal_line = [(0, 0), (1, 0), (2, 0), (3, 0)]
    plus = [(0, 1), (1, 1), (2, 1), (1, 0), (1, 2)]
    reverse_l = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    vertical_line = [(0, 0), (0, 1), (0, 2), (0, 3)]
    bloc = [(0, 0), (1, 0), (0, 1), (1, 1)]

    # rocks = [np.asarray(a) for a in [horizontal_line, plus, reverse_l, vertical_line, bloc]]
    rocks_ = [horizontal_line, plus, reverse_l, vertical_line, bloc]
    rock_names = ["horizontal_line", "plus", "reverse_l", "vertical_line", "bloc"]
    rocks = [(n, rock)for n, rock in zip(rock_names, rocks_)]
    walls = {-1, 7}
    m = 0
    max_y = 0
    shape = set()
    down = lambda p: (p[0], p[1] - 1)

    wind = 0
    past_movements = []
    wind_mod = len(movements)
    rock_n = 0
    rock_mod = len(rocks)
    while 1:
        name, rock = rocks[rock_n % rock_mod]
        rock_n += 1
        # for name, rock in cycle(rocks):
        # for name, rock in cycle(rocks):
        rock = apply(rock, lambda p: center(p, name))
        out_of_bounds = 3 + max_y
        # + (1 if name in ["horizontal_line", "plus", "reverse_l"] else 0)
        up = lambda p: (p[0], p[1] + out_of_bounds)
        rock = apply(rock, up)
        y_min = min([p[1] for p in rock])
        assert y_min - max_y == 3
        breaker = False
        while not breaker:
            movement = movements[wind % wind_mod]
            past_movements += movement
            # for movement in movements:
            if movement == ">":
                sideways = lambda p: (p[0] + 1, p[1])
            else:
                sideways = lambda p: (p[0] - 1, p[1])
            before_wall = rock
            rock = apply(rock, sideways)
            for x, y in rock:
                if x in walls or (x, y) in shape:
                    rock = before_wall
                # elif (x, y) in shape:
                #    rock = before_wall
            before_down = rock
            rock = apply(rock, down)
            for x, y in rock:
                if y == -1:
                    rock = before_down
                    breaker = True
                    break
                elif (x, y) in shape:
                    rock = before_down
                    breaker = True
                    break
            wind += 1

        for coords in rock:
            if coords[1] in shape:
                raise ValueError(f"Duplicate y: {coords}")
            shape.add(coords)

        if is_test:
            if m == 0:
                assert rock == [(2, 0), (3, 0), (4, 0), (5, 0)]
            if m == 1:
                assert rock == [(2, 2), (3, 2), (4, 2), (3, 1), (3, 3)]
            if m == 2:
                assert rock == [(0, 3), (1, 3), (2, 3), (2, 4), (2, 5)]
            if m == 3:
                assert rock == [(4, 3), (4, 4), (4, 5), (4, 6)]
            if m == 4:
                assert rock == [(4, 7), (5, 7), (4, 8), (5, 8)]
            if m == 5:
                assert rock == [(1, 9), (2, 9), (3, 9), (4, 9)]

        max_y = max([y for _, y in shape]) + 1
        m += 1

        if not has_found_offset:
        # if m > 30 and not has_found_offset:
            _max_x = max([p[0] for p in shape])
            display = np.zeros((max_y, _max_x + 2), dtype=bool)

            for x, y in shape:
                display[y, x] = 1

            d = np.rot90(display, 1)
            d = np.rot90(d, 1)
            d = np.fliplr(d)
            # xs = d[:5000:,] 1715?
            xs = d[:500:,]  # 1715?
            asda = []
            for asdsad in xs:
                asda.append("".join([f"%d" % (1 if d else 0) for d in asdsad]))
            agg_x = [ba2int(bitarray(xx)) for xx in asda]
            # for xa in xsb:
            #     a = "!"
            #     b = a
            # ys = d[:,:3000]  # 1715?
            # sum_x = xs.sum(axis=0).tolist()
            # sum_y = xs.sum(axis=1).tolist()
            # canon_move = [d for d in past_movements[-500:]]

            key = "|".join([str(int(d)) for d in agg_x] + [name])  # + canon_move)
            iters[key].add(m)
            # key = tuple(sum_x + [name])
            if key in last_seen:
                cur = iters[key]

                offset = m - last_seen[key][0]
                y_diff = max_y - last_seen[key][1]
                print(f"Found a duplicate at {m}")
                print(f"Last seen at {last_seen[key][0]}")
                print(f"Diff offset: {offset}")
                print(f"Diff Y: {y_diff}")
                factor_mod = (1000000000000 - m) // offset
                # factor_mod -= 2
                y_diff_k = (y_diff * factor_mod)
                max_y += y_diff_k
                m += (factor_mod * offset)
                has_found_offset = True
                shape = set(apply(shape, lambda p: (p[0], p[1] + y_diff_k)))
            last_seen[key] = (m, max_y)


        # if 5 <= m < 100:
        #     d = np.rot90(display, 1)
        #     d = np.rot90(d, 1)
        #     d = np.fliplr(d)
        #     ys = [y for y in range(d.shape[0]) if np.any(d[y, :])]
        #     for y in ys:
        #         print("".join(["#" if d[y, x] else "." for x in range(d.shape[1])]))
        #
        #     print("\n------------------\n")

        if m == n - 1:
            break

    # test
    ## 1514285714288
    ## 1514285714288
    # 1580758017507
    # 1580758017507
    # 1580758017507
    # 1580758017507
    return abs(max_y)


def solve_1(input_=None):
    """
    test=58
    expect=1598415
    """
    is_test = 1 if "test" in input_ else 0

    seed = []
    seed_set = set()
    seed_dict = {}
    seed_counter = Counter()
    seed_ordered_dict = OrderedDict()
    seed_np = np.array([])
    seed_np_2d = np.array([[]])
    seed_default_dict = defaultdict(list)

    area = 0

    with open(locate(input_), "r") as fp:
        lines = list(read_lines(fp)[0])

    return fall(lines, 2023, bool(is_test))


def solve_2(input_=None):
    """
    test=34
    expect=3812909
    """
    is_test = 1 if "test" in input_ else 0

    with open(locate(input_), "r") as fp:
        lines = list(read_lines(fp)[0])

    return fall_p2(lines, 1000000000000, bool(is_test))


if __name__ == "__main__":
    """
    PWSH
    while ($true) {python -m aoc.year_2022.day_03.solve -poll; start-sleep -seconds 1} 

    BASH
    watch -n 3 python3 -m aoc.year_2021.day_03.day -p
    """
    poll_printer = PollPrinter(solve_1, solve_2, challenge_solve_1, challenge_solve_2, test_input, puzzle_input)
    poll_printer.run(sys.argv)

