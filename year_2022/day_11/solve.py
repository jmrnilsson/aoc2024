import math
import math
import re
import sys
from enum import Enum
from functools import reduce
from typing import Tuple, List

from more_itertools import chunked

from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter

# ICE
_default_puzzle_input = "year_2022/day_11/puzzle.txt"
_default_test_input = "year_2022/day_11/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

challenge_solve_1 = 10605
challenge_solve_2 = 2713310158


class Monkey(object):
    def __init__(self, name: int, items: List[int], operation, divisor: int, targets: Tuple[int, int]):
        self.name: int = name
        self.operation = operation
        self.items: List[int] = items
        self.inspections = 0
        self.test_divisor: int = divisor
        self.targets: Tuple[int, int] = targets
        self.other_monkeys = None
        self.lcm = None
        self.custom_divisor = None

    def __repr__(self):
        return self.name

    def throw_at(self, item):
        self.items.append(item)

    def befriend(self, other_monkeys):
        self.other_monkeys = other_monkeys

    def add_lcm(self, _lcm):
        self.lcm = _lcm

    def business(self):
        for item in self.items:
            self.inspections += 1
            worry_level = self.operation(item)
            worry_modulus = worry_level % self.lcm

            if worry_level > self.lcm * 2:
                worry_level = self.lcm + worry_modulus

            if self.custom_divisor:
                worry_level //= self.custom_divisor

            if worry_level // self.test_divisor == worry_level / self.test_divisor:
                self.other_monkeys[self.targets[0]].throw_at(worry_level)
            else:
                self.other_monkeys[self.targets[1]].throw_at(worry_level)

        self.items.clear()


def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


class Day11Operation(Enum):
    POW2 = r"new = old \* old"
    MUL = r"new = old \* (\d+)"
    ADD = r"new = old \+ (\d+)"


def parse_monkey(monkey: List[str]):
    name = int(re.findall(r"(\d+)", monkey[0])[0])
    items = [int(i) for i in re.findall(r"(\d+)", monkey[1])]
    operation = None
    for operation_match in list(Day11Operation):
        if match := re.findall(operation_match.value, monkey[2]):
            if operation_match == Day11Operation.POW2:
                operation = lambda old: math.pow(old, 2)
            elif operation_match == Day11Operation.MUL:
                operation = lambda old: old * int(match[0])
            elif operation_match == Day11Operation.ADD:
                operation = lambda old: old + int(match[0])
            break
    assert operation
    divisor = int(re.findall(r"(\d+)", monkey[3])[0])
    targets = int(re.findall(r"(\d+)", monkey[4])[0]), int(re.findall(r"(\d+)", monkey[5])[0])
    return Monkey(name, items, operation, divisor, targets)


def do_monkey_business(monkeys: List[Monkey], n: int):
    divisors = []
    for monkey in monkeys:
        divisors.append(monkey.test_divisor)
        monkey.befriend(monkeys)

    _lcm = reduce(lcm, divisors)

    for monkey in monkeys:
        monkey.add_lcm(_lcm)

    for n in range(0, n):
        for monkey in monkeys:
            monkey.business()

    inspections = [m.inspections for m in monkeys]
    inspections.sort(key=lambda x: x, reverse=True)
    return inspections[0] * inspections[1]


def solve_1(input_=None):
    """
    test=10605
    expect=54253
    """
    monkeys: List[Monkey] = []

    with open(locate(input_), "r") as fp:
        for monkey in chunked(read_lines(fp), 6):
            monkey_object = parse_monkey(monkey)
            monkeys.append(monkey_object)
            monkey_object.custom_divisor = 3

    return do_monkey_business(monkeys, 20)


def solve_2(input_=None):
    """
    test=2713310158
    expect=13119526120
    """
    monkeys: List[Monkey] = []

    with open(locate(input_), "r") as fp:
        for monkey in chunked(read_lines(fp), 6):
            monkey_object = parse_monkey(monkey)
            monkeys.append(monkey_object)

    return do_monkey_business(monkeys, 10_000)


if __name__ == "__main__":
    """
    PWSH
    while ($true) {python -m aoc.year_2022.day_03.solve -poll; start-sleep -seconds 1} 

    BASH
    watch -n 3 python3 -m aoc.year_2021.day_03.day -p
    """
    poll_printer = PollPrinter(solve_1, solve_2, challenge_solve_1, challenge_solve_2, test_input, puzzle_input)
    poll_printer.run(sys.argv)
