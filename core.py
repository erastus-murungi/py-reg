from dataclasses import dataclass
from itertools import product, chain, count
from string import ascii_uppercase
from typing import ClassVar, Final
from functools import cache

Symbol = str


def yield_letters():
    it = map(
        lambda t: "".join(t),
        chain.from_iterable((product(ascii_uppercase, repeat=i) for i in range(10))),
    )
    _ = next(it)
    yield from it


class State:
    state_ids: ClassVar = count(-1)

    def __init__(self, *, is_start=False, accepts=False):
        self.is_start = is_start
        self.accepts = accepts
        self.id = self.get_id()

    def get_id(self):
        return next(self.state_ids)

    def __repr__(self):
        return f"{self.id}"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id

    def is_empty(self):
        return self is EMPTY_STATE


EMPTY_STATE: Final[State] = State(is_start=False, accepts=False)


class DFAState(State):
    labels_gen = yield_letters()

    def __init__(self, *, is_start=False, accepts=False, from_states):
        self.from_states = from_states
        super().__init__(is_start=is_start, accepts=accepts)

    def get_id(self):
        return self.get_label(self.from_states)

    @staticmethod
    @cache
    def get_label(frozen):
        return next(DFAState.labels_gen)
