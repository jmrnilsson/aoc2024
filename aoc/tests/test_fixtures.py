from dataclasses import dataclass
from typing import Callable, Tuple, Generator, List
from docstring_parser import parse as parse_docs, DocstringMeta, Docstring


@dataclass
class AdventFixture:
    challenge: str
    expect: str

    def __init__(self, challenge: DocstringMeta, expect: DocstringMeta):
        self.challenge = challenge.description
        self.expect = expect.description


def make_fixture(fn: Callable) -> AdventFixture:
    doc_str_1: Docstring = parse_docs(fn.__doc__)
    challenge: DocstringMeta = next(filter(lambda m: m.args[0] == "challenge", doc_str_1.meta))
    expect: DocstringMeta = next(filter(lambda m: m.args[0] == "expect", doc_str_1.meta))
    return AdventFixture(challenge, expect)


def _get_meta_description_named_challenge(fn: Callable) -> int:
    doc_string: Docstring = parse_docs(fn.__doc__)
    for meta in doc_string.meta:
        if meta.args[0] == "challenge":
            return int(meta.description)


def get_challenges_from_meta(fn1: Callable, fn2: Callable | None) -> List[int]:
    challenges = [_get_meta_description_named_challenge(fn1)]

    if fn2:
        challenges.append(_get_meta_description_named_challenge(fn2))

    return challenges
