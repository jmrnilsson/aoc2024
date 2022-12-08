import json
import logging
import re
import sys
import traceback
from collections import Counter, OrderedDict
from functools import wraps
from time import perf_counter
from typing import List

import numpy as np

_logger = logger = logging.getLogger(__name__)


def _timing(*args, **kw):
    # https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
    func, args_ = args[0], args[1:]
    ts = perf_counter()
    result = func(*args_, **kw)
    te = perf_counter()
    input_file = re.sub("^.*aoc", "", args_[0])
    input_file = re.sub(r"\\", " ", input_file)
    print('%s: %s %s %2.4f ms' % (func.__name__, input_file, kw, (te - ts) * 1000))
    return result


def puzzle_nocolor(result):
    return print(f"PUZZLE: {result}")


def test_nocolor(result):
    return print(f"TEST: {result}")


def solve_1_timed(solve_1, puzzle_input=None):
    return _timing(solve_1, puzzle_input)


def solve_2_timed(solve_2, puzzle_input=None):
    return _timing(solve_2, puzzle_input)


class PollPrinter(object):

    def __init__(self, solve_1, solve_2, challenge_1, challenge_2, test_input, puzzle_input):
        self.solve_1 = solve_1
        self.solve_2 = solve_2
        self.challenge_1 = challenge_1
        self.challenge_2 = challenge_2
        self.test_input = test_input
        self.puzzle_input = puzzle_input

    def run(self, argv: List[str]):
        args = argv[1:]
        if not args:
            self.print_timed()
        elif re.match("^-poll$|^-p$", args[0]):
            self.poll_print()
        elif re.match("^-json1$|^-j1$", args[0]):
            self.poll_json_1()
        elif re.match("^-json2$|^-j2$", args[0]):
            self.poll_json_2()

    def poll_json_1(self):
        test = self.solve_1(self.test_input)
        print(json.dumps({
            "test": test,
            "puzzle": self.solve_1(self.puzzle_input),
            "ok": test == self.challenge_1
        }))

    def poll_json_2(self):
        test = self.solve_2(self.test_input)
        print(json.dumps({
            "test": test,
            "puzzle": self.solve_2(self.puzzle_input),
            "ok": test == self.challenge_2
        }))

    def poll_print(self):
        test_nocolor(solve_1_timed(self.solve_1, self.test_input))
        puzzle_nocolor(solve_1_timed(self.solve_1, self.puzzle_input))

        try:
            Printer.div()
            test_nocolor(solve_2_timed(self.solve_2, self.test_input))
            puzzle_nocolor(solve_2_timed(self.solve_2, self.puzzle_input))
            Printer.fin()
        except Exception as exc:
            print(f"No solve part 2. Type={type(exc)}")

    def print_timed(self):
        Printer.test(solve_1_timed(self.solve_1, self.test_input))
        Printer.puzzle(solve_1_timed(self.solve_1, self.puzzle_input))

        try:
            Printer.div()
            Printer.test(solve_2_timed(self.solve_2, self.test_input))
            Printer.puzzle(solve_2_timed(self.solve_2, self.puzzle_input))
        except Exception as exc:
            traceback.print_exc(file=sys.stdout)

        Printer.fin()


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Printer(object):

    # noinspection PyBroadException
    @staticmethod
    def _print(result, color):
        try:
            result_str_arg = []
            can_materialize_to_list = bool(isinstance(result, (list, tuple)))
            if can_materialize_to_list:
                result_list = list(result)
                result_str_arg.append(str(result_list.pop(0)).strip())

                for ancillary_result in result_list:
                    ancillary_str = str(ancillary_result)

                    if len(ancillary_str) > 200:
                        result_str_arg.append(ancillary_str[:200] + "...")
                    else:
                        result_str_arg.append(ancillary_str)

                result_str = "  ".join(result_str_arg)
                # print(result_str)
                print(f"{color}", result_str, f"{BColors.ENDC}")
            else:
                # print(result)
                print(f"{color}", result, f"{BColors.ENDC}")
        except Exception as e:
            _logger.error(e)
            # print(result)
            print(f"{color}", result, f"{BColors.ENDC}")

    @staticmethod
    def puzzle(result):
        return Printer._print(result, BColors.OKGREEN)

    @staticmethod
    def test(result):
        return Printer._print(result, BColors.WARNING)

    @staticmethod
    def tes2(result):
        return Printer._print(result, BColors.OKBLUE)

    @staticmethod
    def fin():
        return print("\n\n")

    @staticmethod
    def div():
        return print(" ---------------------------------")
