import sys
from typing import Tuple, Set, List, Dict

from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter

# ICE
_default_puzzle_input = "year_2022/day_01/puzzle.txt"
_default_test_input = "year_2022/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

challenge_solve_1 = 13
challenge_solve_2 = "exception"


def move(direction, position, amount, limit=None):
    limited_amount = min(amount, limit) if limit else amount
    if direction == "N":
        return position[0], position[1] + limited_amount
    if direction == "S":
        return position[0], position[1] - limited_amount
    if direction == "E":
        return position[0] + limited_amount, position[1]
    if direction == "W":
        return position[0] - limited_amount, position[1]
    raise ValueError(f"Cannot go to there {direction}")


def follow(previous, current):
    y, x = previous
    tail_y, tail_x = current
    if tail_y not in [y, y + 1, y - 1] or tail_x not in [x, x + 1, x - 1]:
        if tail_y < y:
            tail_y += 1
        elif tail_y > y:
            tail_y -= 1
        if tail_x < x:
            tail_x += 1
        elif tail_x > x:
            tail_x -= 1

    return tail_y, tail_x


def solve_1(input_=None):
    """
    test=13
    expect=6314
    """
    # test = 1 if "test" in input_ else 0
    lines: List[Tuple[str, str, int]] = []
    position: Tuple[int, int] = 0, 0
    tail_position: Tuple[int, int] = position
    trail: Set[Tuple[int, int]] = {tail_position}
    translation: Dict[str, str] = {"U": "N", "D": "S", "L": "W", "R": "E"}

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            silly_direction, amount = line.split(" ")
            proper_direction = translation[silly_direction]
            lines.append((proper_direction, silly_direction, int(amount)))

    # n = 0
    for direction, silly_direction, amount in lines:
        destination = move(direction, position, amount)

        while position != destination:
            position = move(direction, position, amount, 1)

            tail_position = follow(position, tail_position)
            trail.add(tail_position)

            # if test:
            #     matrix_out = np.zeros((6, 5))
            #     matrix_out[position] = 1
            #     matrix_out[tail_position] = 2
            #     tr = np.transpose(matrix_out)
            #     tr = np.flip(tr, 0)
            #     print(f"{n}: {silly_direction} {amount}\n{tr}\n")
            #     n += 1

    return sum(1 for _ in trail)


def solve_2(input_=None):
    """
    test=36
    expect=2504
    """
    lines = []
    position: Tuple[int, int] = 0, 0
    tail_positions: List[Tuple[int, int]] = [position for _ in range(0, 9)]
    assert len(tail_positions) == 9
    trail: Set[Tuple[int, int]] = set(tail_positions)
    translation: Dict[str, str] = {"U": "N", "D": "S", "L": "W", "R": "E"}

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            silly_direction, amount = line.split(" ")
            proper_direction = translation[silly_direction]
            lines.append((proper_direction, silly_direction, int(amount)))

    for direction, _, amount in lines:
        destination = move(direction, position, amount)

        while position != destination:
            position = move(direction, position, amount, 1)

            previous = position
            materialized = list(tail_positions)
            tail_positions.clear()

            for t in range(0, 9):
                previous = follow(previous, materialized[t])
                tail_positions.append(previous)

                if t == 8:
                    trail.add(previous)

    return sum(1 for _ in trail)


if __name__ == "__main__":
    """
    PWSH
    while ($true) {python -m aoc.year_2022.day_03.solve -poll; start-sleep -seconds 1} 

    BASH
    watch -n 3 python3 -m aoc.year_2021.day_03.day -p
    """
    poll_args = solve_1, solve_2, challenge_solve_1, challenge_solve_2, test_input, puzzle_input
    poll_printer = PollPrinter(*poll_args, test_input_2=test_input_2)
    poll_printer.run(sys.argv)

