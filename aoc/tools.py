import itertools
import re
from collections.abc import Callable
from collections import OrderedDict
import functools
from typing import List

from aoc.helpers import Printer

def pop(iterable: List, *_args: int):
    args = sorted(_args, reverse=True)
    for a in args:
        iterable.pop(a)

def group_by(iterable, key):
    sorted_iterable = sorted(iterable, key=key)
    for k, v in itertools.groupby(sorted_iterable, key):
        yield k, list(v)


def _find_vectors(matrix, vector_matcher: Callable = None, compress=True):
    vectors = []
    _vector_matcher = vector_matcher if vector_matcher else lambda ch: re.match(r"\w", ch)
    for y in range(len(matrix)):
        last_x, n, x_values = None, 0, []
        for x in range(len(matrix[0])):
            value = matrix[y][x]
            if _vector_matcher(value):
                if last_x or n == 0:
                    x_values.append((y, x, value))
            else:
                if len(x_values) > 1:
                    vectors.append(x_values)
                x_values = []

        if len(x_values) > 1:
            vectors.append(x_values)

    for x in range(len(matrix[0])):
        last_y, n, y_values = None, 0, []
        for y in range(len(matrix)):
            value = matrix[y][x]
            if _vector_matcher(value):
                if last_y or n == 0:
                    y_values.append((y, x, value))
            else:
                if len(y_values) > 1:
                    vectors.append(y_values)
                y_values = []
        if len(y_values) > 1:
            vectors.append(y_values)

    if not compress:
        return vectors
    else:
        compressed = []
        for vector in vectors:
            first, last = vector[0][:2], vector[-1][:2]
            values = "".join(map(lambda v: v[2], vector))
            compressed.append((first, last, values))
        return compressed


def to_matrix(jagged_input: List[str], pad_char: str, replace_chars_: str = None):
    """
    Might be broken for Python >3.9.
    """
    def replace(c):
        return pad_char if c in replace_chars else c

    replace_chars = replace_chars_ or ""
    _matrix = [[replace(char) for char in list(row)] for row in jagged_input]
    return matrix_pad(_matrix, pad_char)


def bitmap(padded: List[List[str]], vector_matcher: Callable = None):
    vectors = list(_find_vectors(padded))

    n, m = 0, 0
    x_len = len(padded[0]) + 4
    y_len = len(padded) + 4
    root = []
    for y in range(y_len):
        x_range = []
        y_indices = str(n - 4).rjust(3) + "|"
        for x in range(x_len):
            if x < 4:
                x_value = y_indices[x]
            else:
                x_value = padded[y - 4][x - 4]
            x_range.append(x_value)
        x_indices = [list(str(d).rjust(3) + "_") for d in range(x_len - 4)]
        if y < 4:
            x_indices_or_value = list("    ") + list(map(lambda x_s: x_s[y], x_indices))
        else:
            x_indices_or_value = x_range
        root.append(x_indices_or_value)
        n += 1
    return root, vectors


def bitmap_print_only(matrix: List[List[str]], vector_matcher: Callable = None):
    root, vectors = bitmap(matrix, vector_matcher)
    str_matrix = ["".join(inner) for inner in root]
    for y in str_matrix:
        print(y)


def bitmap_print_matrix(matrix: List[List[str]], vector_matcher: Callable = None):
    root, vectors = bitmap(matrix, vector_matcher)
    str_matrix = ["".join(inner) for inner in root]
    for y in str_matrix:
        print(y)

    Printer.div()

    for vec in vectors:
        print(vec)

    Printer.div()

    if vectors:
        print("vectors = [")
        for n, vec in enumerate(vectors):
            from_, to, values = vec
            optional_comma = "," if n + 1 != len(vectors) else ""
            print(f"    tools.vector({from_}, {to}, matrix){optional_comma}")
        print("]")


def bitmap_print(jagged_input: List[str], pad_char: str, replace_chars=None, vector_matcher: Callable = None):
    padded = to_matrix(jagged_input, pad_char, replace_chars)
    bitmap_print_matrix(padded, vector_matcher)


def vector(y_x0, y_x1, matrix, str_join=False):
    values = []
    y_0, x_0 = y_x0
    y_1, x_1 = y_x1
    if y_0 == y_1:
        for x in range(x_0, x_1 + 1):
            values.append(matrix[y_0][x])
    elif x_0 == x_1:
        for y in range(y_0, y_1 + 1):
            values.append(matrix[y][x_0])
    else:
        raise ValueError("Vector not multi-dimensional")

    if str_join:
        return "".join(values)

    return values


def get_vectors(matrix: List[List[str]], vector_matcher: Callable = None, str_join=False):
    result = []
    vectors = _find_vectors(matrix, vector_matcher)
    for from_, to, _ in vectors:

        result.append(vector(from_, to, matrix, str_join=str_join))

    return result


def transpose(matrix):
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]


def matrix_pad(matrix, pad_char):
    """
    Pad right in nested array in matrix.
    """
    max_len = max([len(li) for li in matrix])
    return [line + [pad_char for _ in range(0, max_len - len(line))] for line in matrix]


def reverse_matrix_x(matrix):
    """
    Reverse nested array in matrix
    """
    return [list(reversed(line)) for line in matrix]


def matrix_str_join_x(matrix):
    """
    String join nested array in matrix.
    """
    return ["".join(line) for line in matrix]


def matrix_copy(matrix):
    """
    Naive copy of matrix assuming equal length.
    :param matrix:
    :return:
    """
    x_len = len(matrix[0])
    y_len = len(matrix)
    root = []
    for y in range(y_len):
        x_range = []
        for x in range(x_len):
            x_range.append(matrix[y][x])
        root.append(x_range)
    return root


def matrix_enumerate(matrix):
    """
    Enumerates x, y and value tuple from a matrix.
    :param matrix:
    :return:
    """
    for y, y_s in enumerate(matrix):
        for x, value in enumerate(y_s):
            yield x, y, value


def matrix_try_get(matrix, x, y, default=None):
    """
    Get value within bounds or default value.
    :param matrix:
    :param x:
    :param y:
    :param default: value instead of raising IndexError
    :return:
    """
    if x < 0 or y < 0 or x >= len(matrix[0]) or y >= len(matrix):
        return default

    try:
        cell = matrix[y][x]
        return cell
    except IndexError:
        return default


def max_min_tuple(iterable):
    max_, min_ = None, None

    for item in iterable:
        if not max_ or item > max_:
            max_ = item
        if not min_ or item < min_:
            min_ = item

    return max_, min_


class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
           not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory, OrderedDict.__repr__(self))


def fold_left(func, acc, xs):
    # print(foldl(operator.sub, 0, [1, 2, 3]))  # -6
    # print(foldl(operator.add, 'L', ['1', '2', '3']))  # 'L123'
    return functools.reduce(func, xs, acc)


def _flip(func):
    @functools.wraps(func)
    def new_func(x, y):
        return func(y, x)
    return new_func


def fold_right(func, acc, xs):
    # print(foldr(operator.sub, 0, [1, 2, 3]))  # 2
    # print(foldr(operator.add, 'R', ['1', '2', '3']))  # '123R'
    return functools.reduce(_flip(func), reversed(xs), acc)
