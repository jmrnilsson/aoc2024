import os
import re
from functools import wraps
import shutil
from time import perf_counter
import logging

import aoc
from aoc.printer import ANSIColors as BColors

_logger = logger = logging.getLogger(__name__)


def read_lines(fp):
    return [re.sub("(\n$)", "", line) for line in fp.readlines() if line != "\n"]


def timing(f):
    # https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
    @wraps(f)
    def wrap(*args, **kw):
        ts = perf_counter()
        result = f(*args, **kw)
        te = perf_counter()
        input_file = re.sub("^.*aoc", "", args[0])
        input_file = re.sub(r"\\", " ", args[0])
        print('%s: %s %s %2.4f ms' % (f.__name__, input_file, kw, (te - ts) * 1000))
        return result
    return wrap


def locate(file_name):
    return os.path.normpath(os.path.join(aoc.ROOT_DIR, file_name))


def build_location(file_, filename):
    return os.path.normpath(str(os.path.join(os.path.dirname(os.path.normpath(file_)), filename)))


def build_location_test(year, day, filename):
    day_zfill = str(day).zfill(2)
    aoc_path = os.path.dirname(os.path.normpath(__file__))
    work_path = os.path.normpath(os.path.join(aoc_path, f"year_{year}/day_{day_zfill}", filename))
    return work_path


def copy_tree(src, dst):
    """
    Distutil copy_tree shim for Python 3.12
    """
    os.makedirs(dst, exist_ok=True)
    for item in os.listdir(src):
        src_item = os.path.join(src, item)
        dst_item = os.path.join(dst, item)
        if os.path.isdir(src_item):
            shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
        else:
            shutil.copy2(src_item, dst_item)


def loc_input(file_, name_ext):
    fr = file_.replace("_part_2", "")
    fr = fr.replace("_part_1", "")
    d = os.path.splitext(os.path.basename(os.path.normpath(fr)))[0]
    # y = os.path.basename(os.path.dirname(os.path.normpath(fr)))
    y = os.path.dirname(os.path.normpath(fr))
    return "{y}/{d}{name_ext}".format(y=y, d=d, name_ext=name_ext)


def loc_input_old(file_, name_):
    day_ = os.path.splitext(os.path.basename(os.path.normpath(file_)))[0]
    path_and_year = os.path.dirname(os.path.normpath(file_))
    return os.path.join(path_and_year, day_ + name_)


def puzzle_nocolor(result):
    return print(f"PUZZLE: {result}")


def test_nocolor(result):
    return print(f"TEST: {result}")


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
