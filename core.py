from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cache
from itertools import product, chain, count
from string import ascii_uppercase
from typing import ClassVar, Final, Collection, MutableMapping
from utils import is_iterable

import graphviz

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

    @staticmethod
    def get_pair():
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
    labels_gen = yield_letters()

    def __init__(self, from_states, *, is_start=False, accepts=False):
        self.sources = from_states
        super().__init__(is_start=is_start, accepts=accepts)

    def get_id(self):
        return self.get_label(self.sources)

    @staticmethod
    @cache
    def get_label(_frozen):
        return next(DFAState.labels_gen)

    def is_null(self):
        return self is NullDfaState


NullDfaState: Final[State] = DFAState(from_states=None)


class FiniteStateAutomaton(
    defaultdict[State, MutableMapping[Symbol, DFAState | list[State]]], ABC
):
    states: set[State] | set[DFAState]
    start_state: State | DFAState
    accept: State | set[DFAState]
    symbols: set[Symbol]

    @abstractmethod
    def transition(self, state: State, symbol: Symbol) -> DFAState | Collection[State]:
        pass

    @abstractmethod
    def all_transitions(self) -> tuple[Symbol, State, State]:
        pass

    @abstractmethod
    def _dict(
        self,
    ) -> defaultdict[State, MutableMapping[Symbol, DFAState | list[State]]]:
        pass

    def draw_with_graphviz(self):
        dot = graphviz.Digraph(
            self.__class__.__name__ + ", ".join(map(str, self.states)),
            format="pdf",
            engine="circo",
        )
        dot.attr("node", shape="circle")

        for symbol, s1, s2 in self.all_transitions():
            if s1.is_start:
                dot.node(
                    str(s1.id),
                    color="green",
                    shape="doublecircle" if s1.accepts else "circle",
                )
            else:
                dot.node(
                    f"{s1.id}",
                    shape="doublecircle" if s1.accepts else "circle",
                )

            dot.node(f"{s2.id}", shape="doublecircle" if s2.accepts else "circle")
            dot.edge(str(s1.id), str(s2.id), label=symbol)

        dot.node("start", shape="none")
        dot.edge("start", f"{self.start_state.id}", arrowhead="vee")
        dot.render(view=True, directory="graphs", filename=str(id(self)))

    def update_states_set(self):
        for source, table in self.items():
            self.states.add(source)
            for sinks in table.values():
                if is_iterable(sinks):
                    for sink in sinks:
                        self.states.add(sink)
                else:
                    self.states.add(sinks)

    def set_start(self, state: State):
        self.start_state = state
        self.start_state.is_start = True

    @abstractmethod
    def gen_dfa_state_set_flags(self, sources) -> DFAState:
        pass
