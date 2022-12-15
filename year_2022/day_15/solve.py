import operator
import re
import sys
from typing import List, Tuple, Set, Callable

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


def manhattan(sensor, beacon):
    return abs(sensor[0] - beacon[0]) + abs(sensor[1] - beacon[1])


def manhattan_traverse_y(start_x: int, end_x: int, items: List[Tuple[int, int]], beacons_in_x: Set[int], sensors_in_x: Set[int], is_test: bool):
    xs = set()
    for x in range(start_x, end_x + 1):
        for sensor_x, distance_in_x in items:
            if sensor_x - distance_in_x <= x <= sensor_x + distance_in_x:
                if x not in beacons_in_x and x not in sensors_in_x:
                    xs.add(x)
                    break

    return sum(1 for _ in xs)


def solve_1(input_=None):
    """
    test=26
    expect=5256611
    """
    is_test = 1 if "test" in input_ else 0
    seed = []

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            digits = [int(i) for i in re.findall(r"-?\d+", line)]
            sensor = tuple(digits[:2])
            beacon = tuple(digits[2:])
            seed.append((sensor, beacon))

    y = 10 if is_test else 2000000
    parameters = []
    for sensor, beacon in seed:
        distance = manhattan(sensor, beacon)
        distance_in_x = distance - abs(sensor[1] - y)
        parameter = sensor[0], sensor[1], beacon[0], beacon[1], distance_in_x
        parameters.append(parameter)

    min_x = min([point[0] - point[4] for point in parameters]) - 1
    max_x = max([point[0] + point[4] for point in parameters]) + 1
    items = [(p[0], p[4]) for p in parameters]
    beacons_in_x = {p[2] for p in parameters if p[3] == y}
    sensors_in_x = {p[0] for p in parameters if p[1] == y}

    return manhattan_traverse_y(min_x, max_x, items, beacons_in_x, sensors_in_x, bool(is_test))


def find_beacon(x, y, seed):
    found = True

    for sensor, beacon in seed:
        distance = abs(sensor[0] - beacon[0]) + abs(sensor[1] - beacon[1])
        distance_in_x = distance - abs(sensor[1] - y)
        if distance_in_x > 0:
            if sensor[0] - distance_in_x <= x <= sensor[0] + distance_in_x:
                found = False
                break

    if found:
        return x * 4000000 + y

    return None


def next_edge(operator_x: Callable, operator_y: Callable, sensor: Tuple[int, int], k, min_k, max_k):
    edge = operator_x(sensor[0], k), operator_y(sensor[1],  k)
    if min_k <= edge[0] <= max_k:
        if min_k <= edge[1] <= max_k:
            return edge
    return None


def every_von_neumann_edge(max_k, min_k, seed: List[Tuple[Tuple[int, int], Tuple[int, int]]]):
    for sensor, beacon in seed:
        distance = abs(sensor[0] - beacon[0]) + abs(sensor[1] - beacon[1]) + 1
        for k in range(0, distance + 1):
            for op_x, op_y in [(operator.sub, operator.add), (operator.add, operator.add)]:
                sensor_up = sensor[0], sensor[1] - distance
                if edge := next_edge(op_x, op_y, sensor_up, k, min_k, max_k):
                    yield edge
            for op_x, op_y in [(operator.sub, operator.sub), (operator.add, operator.sub)]:
                sensor_down = sensor[0], sensor[1] + distance
                if edge := next_edge(op_x, op_y, sensor_down, k, min_k, max_k):
                    yield edge


def solve_2(input_=None):
    """
    test=56000011
    expect=13337919186981
    """
    is_test = 1 if "test" in input_ else 0
    seed: List[Tuple, Tuple] = []

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            digits = [int(i) for i in re.findall(r"-?\d+", line)]
            sensor = tuple(digits[:2])
            beacon = tuple(digits[2:])
            seed.append((sensor, beacon))

    min_k = 0
    max_k = 4000000 if not is_test else 20

    for x, y in every_von_neumann_edge(max_k, min_k, seed):
        if find_beacon(x, y, seed):
            if 0 <= x <= max_k:
                if 0 <= y <= max_k:
                    print(f"y: {y} x: {x}")
                    return x * 4000000 + y

    return None



if __name__ == "__main__":
    """
    PWSH
    while ($true) {python -m aoc.year_2022.day_03.solve -poll; start-sleep -seconds 1} 

    BASH
    watch -n 3 python3 -m aoc.year_2021.day_03.day -p
    """
    poll_printer = PollPrinter(solve_1, solve_2, challenge_solve_1, challenge_solve_2, test_input, puzzle_input)
    poll_printer.run(sys.argv)

