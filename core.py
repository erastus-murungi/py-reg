from abc import ABC, abstractmethod
from collections import defaultdict
from enum import IntFlag, auto
from functools import cache
from itertools import chain, count, product
from string import ascii_uppercase
from typing import (ClassVar, Collection, Final, NamedTuple,
                    Optional)

import graphviz


class RegexFlag(IntFlag):
    NOFLAG = auto()
    IGNORECASE = auto()
    MULTILINE = auto()
    DOTALL = auto()  # make dot match newline


class Matchable(ABC):
    @abstractmethod
    def match(self, text: str, position: int, flags: RegexFlag) -> bool:
        ...


def yield_letters():
    it = map(
        lambda t: "".join(t),
        chain.from_iterable((product(ascii_uppercase, repeat=i) for i in range(10))),
    )
    _ = next(it)
    yield from it


class State:
    ids: ClassVar = count(-1)

    def __init__(self, *, is_start=False, accepts=False, lazy=False):
        self.is_start = is_start
        self.accepts = accepts
        self.id = self._gen_id()
        self.lazy = lazy

    def _gen_id(self):
        return next(self.ids)

    @staticmethod
    def pair():
        return State(), State()

    def __repr__(self):
        return f"{self.id}"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id

    def is_null(self):
        return self is NullState


NullState: Final[State] = State(is_start=False, accepts=False)


class DFAState(State):
    _labels_gen = yield_letters()

    def __init__(self, from_states, *, is_start=False, accepts=False):
        self.sources = from_states
        super().__init__(is_start=is_start, accepts=accepts)

    def _gen_id(self):
        return self._gen_label(self.sources)

    @staticmethod
    @cache
    def _gen_label(_frozen):
        return next(DFAState._labels_gen)

    def is_null(self):
        return self is NullDfaState


NullDfaState: Final[State] = DFAState(from_states=None)


class Transition(NamedTuple):
    matchable: Matchable
    end: State

    def match(
        self, text: str, position: int, flags: RegexFlag, default=None
    ) -> Optional[State]:
        if self.matchable.match(text, position, flags):
            return self.end
        return default


class FiniteStateAutomaton(
    defaultdict[State, set[Transition]],
    ABC,
):
    states: set[State]
    start_state: State | DFAState
    accept: State | set[DFAState]
    symbols: set[Matchable]

    @abstractmethod
    def transition(
        self, state: State, symbol: Matchable
    ) -> DFAState | Collection[State]:
        pass

    @abstractmethod
    def all_transitions(self) -> tuple[Matchable, State, State]:
        pass

    def graph(self):
        dot = graphviz.Digraph(
            self.__class__.__name__ + ", ".join(map(str, self.states)),
            format="pdf",
            engine="circo",
        )
        dot.attr("graph", rankdir="LR")

        seen = set()

        for symbol, s1, s2 in self.all_transitions():
            if s1 not in seen:
                if s1.is_start:
                    dot.node(
                        str(s1.id),
                        color="green",
                        shape="doublecircle" if s1.accepts else "circle",
                        style="filled",
                    )
                elif s1.lazy:
                    dot.node(
                        str(s1.id),
                        color="red",
                        shape="doublecircle" if s1.accepts else "circle",
                        style="filled",
                    )
                else:
                    dot.node(
                        f"{s1.id}",
                        shape="doublecircle" if s1.accepts else "circle",
                    )
                seen.add(s1)
            if s2 not in seen:
                seen.add(s2)
                if s2.lazy:
                    dot.node(
                        f"{s2.id}",
                        color="gray",
                        style="filled",
                        shape="doublecircle" if s2.accepts else "circle",
                    )
                else:
                    dot.node(
                        f"{s2.id}", shape="doublecircle" if s2.accepts else "circle"
                    )
            dot.edge(str(s1.id), str(s2.id), label=str(symbol))

        dot.node("start", shape="none")
        dot.edge("start", f"{self.start_state.id}", arrowhead="vee")
        dot.render(view=True, directory="graphs", filename=str(id(self)))

    def update_states_set(self):
        for _, start, end in self.all_transitions():
            self.states.update({start, end})

    def _dict(self) -> defaultdict[State, set[Transition]]:
        d = defaultdict(set)
        for state, transitions in self.items():
            d[state] = transitions.copy()
        return d

    def set_start(self, state: State):
        self.start_state = state
        self.start_state.is_start = True

    @abstractmethod
    def gen_dfa_state_set_flags(self, sources) -> DFAState:
        pass
