import math
from typing import Callable, List, Tuple, Any

import more_itertools


class Box:
    _value: int

    def __init__(self, value: int):
        self._value = value

    def set(self, value: int):
        self._value = value

    def get(self):
        return self._value

    def __repr__(self):
        return str(self._value)

class RaiseBox(Box):

    def get(self):
        raise TypeError("7 can never happen")


class ChronospatialComputer:
    a: Box
    b: Box
    c: Box
    operator: str
    combo_operand: Box
    literal_operand: int
    i: int
    optcodes: List[Tuple[Any, ...]]

    def __init__(self, a: int, b: int, c: int, optcodes, pt: Callable | None = None):
        self.pt = pt or print
        self.a = Box(a)
        self.b = Box(b)
        self.c = Box(c)
        self.combo_operand = Box(-1)
        self.stay = False
        self.raw_optcodes = list(optcodes)
        self.optcodes = list(more_itertools.chunked(optcodes, 2))
        self.i = -1
        self.tail_recursions = [self.i]
        self.warp_index_to_i_on_next = None

    def set_operand(self, raw_operand):
        match raw_operand:
            case 4:
                self.combo_operand.set(self.a.get())
            case 5:
                self.combo_operand.set(self.b.get())
            case 6:
                self.combo_operand.set(self.c.get())
            case 7:
                self.combo_operand = RaiseBox(-1)
            case _:
                self.combo_operand.set(raw_operand)

        self.literal_operand = raw_operand

    def set_operation(self, optcode: int):
        match optcode:
            case 0:
                self.operator = "adv"
            case 1:
                self.operator = "bxl"
            case 2:
                self.operator = "bst"
            case 3:
                self.operator = "jnz"
            case 4:
                self.operator = "bxc"
            case 5:
                self.operator = "out"
            case 6:
                self.operator = "bdv"
            case 7:
                self.operator = "cdv"

    def apply(self):
        # print(f"operator: {self.operator}")
        getattr(self, self.operator)()

    def next(self):
        if not self.stay:
            self.i += 1
        try:
            return self.optcodes[self.i]
        except IndexError:
            return None
        finally:
            self.stay = False

    def _dv(self):
        ob = self.combo_operand.get()
        floored =  math.floor(self.a.get() / math.pow(2, ob))
        return floored

    def adv(self):
        self.a.set(self._dv())

    def bxl(self):
        self.b.set(self.b.get() ^ self.literal_operand)

    def bst(self):
        self.b.set(self.combo_operand.get() % 8)

    def jnz(self):
        if self.a.get() == 0:
            return

        if self.literal_operand == 3:
            self.stay = True

        self.i = self.literal_operand - 1
        print(f"A: {self.a.get()}")


    def bxc(self):
        self.b.set(self.b.get() ^ self.c.get())
        # Ignores input

    def out(self):
        combo = self.combo_operand.get()
        result = combo % 8
        self.pt(result)

    def bdv(self):
        self.b.set(self._dv())

    def cdv(self):
        self.c.set(self._dv())
