import re
import re
import sys

from aoc import tools
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter

# ICE
_default_puzzle_input = "year_2022/day_05/puzzle.txt"
_default_test_input = "year_2022/day_05/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

challenge_solve_1 = "exception"
challenge_solve_2 = "exception"


def stack_inverted(input_, start):
    head = []
    with open(locate(input_), "r") as fp_0:
        head_lines = list(read_lines(fp_0))[:start]
        for line in head_lines:
            if not line.strip():
                continue

            plain_line = line.replace("[", " ").replace("]", " ").replace("\n", "")
            head.append(list(plain_line))

    matrix_padded = tools.matrix_pad(head, " ")
    transpose_head = tools.transpose(matrix_padded)
    reverse_matrix_head = tools.reverse_matrix_x(transpose_head)
    stacks = {int(line[0]): [li for li in line[1:] if li.strip()] for line in reverse_matrix_head if "".join(line).strip()}
    return stacks


def solve_1(input_=None):
    start = 4 if "test" in input_ else 9
    crates = stack_inverted(input_, start)

    step = 0
    with open(locate(input_), "r") as fp:
        lines = [l.strip() for l in read_lines(fp) if l.strip()][start:]
        for line in lines:
            n, from_, to = [int(d) for d in re.findall(r'\d+', line)]
            while n > 0:
                crates[to].append(crates[from_].pop())
                n -= 1
            step += 1

    return str.join("", [li[-1] for li in crates.values()])


def solve_2(input_=None):
    start = 4 if "test" in input_ else 9
    crates = stack_inverted(input_, start)

    step = 0
    with open(locate(input_), "r") as fp:
        lines = [l.strip() for l in read_lines(fp) if l.strip()][start:]
        for line in lines:
            n, from_, to = [int(d) for d in re.findall(r'\d+', line)]
            swap = []
            while 1:
                if n < 1:
                    break

                swap.insert(0, crates[from_].pop())
                n -= 1

            for s in swap:
                crates[to].append(s)

            step += 1

    return str.join("", [li[-1] for li in crates.values()])


if __name__ == "__main__":
    """
    PWSH
    while ($true) {python -m aoc.year_2022.day_03.solve -poll; start-sleep -seconds 1} 

    BASH
    watch -n 3 python3 -m aoc.year_2021.day_03.day -p
    """

    poll_printer = PollPrinter(solve_1, solve_2, challenge_solve_1, challenge_solve_2, test_input, puzzle_input)
    args = sys.argv[1:]
    if not args:
        poll_printer.print_timed()
    elif re.match("^-poll$|^-p$", args[0]):
        poll_printer.poll_print()
    elif re.match("^-json1$|^-j1$", args[0]):
        poll_printer.poll_json_1()
    elif re.match("^-json2$|^-j2$", args[0]):
        poll_printer.poll_json_2()


