from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cache
from itertools import chain, count, product
from string import ascii_uppercase
from typing import (ClassVar, Collection, Final, Generic, MutableMapping,
                    Optional, Sequence, TypeVar)

import graphviz

from utils import Comparable, is_iterable

T = TypeVar("T", bound=Comparable)


class MatchableMixin(Generic[T], ABC):
    @abstractmethod
    def match(self, position: int, text: Sequence[T]) -> bool:
        ...


class CompoundMatchableMixin(MatchableMixin, ABC):
    pass


def yield_letters():
    it = map(
        lambda t: "".join(t),
        chain.from_iterable((product(ascii_uppercase, repeat=i) for i in range(10))),
    )
    _ = next(it)
    yield from it


class State:
    ids: ClassVar = count(-1)

    def __init__(self, *, is_start=False, accepts=False):
        self.is_start = is_start
        self.accepts = accepts
        self.id = self._gen_id()

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
        return self.get_label(self.sources)

    @staticmethod
    @cache
    def get_label(_frozen):
        return next(DFAState._labels_gen)

    def is_null(self):
        return self is NullDfaState


NullDfaState: Final[State] = DFAState(from_states=None)


class SymbolDispatchedMapping(MutableMapping):
    def __init__(self, default=None):
        self.default_factory = default
        self.sorted_map: list[tuple[CompoundMatchableMixin, set[State] | State]] = []
        if default is None:
            self.hash_map = {}
        else:
            self.hash_map: defaultdict[
                MatchableMixin, set[State] | State
            ] = defaultdict(default)

    def __setitem__(self, symbol, v) -> None:
        if isinstance(symbol, CompoundMatchableMixin):
            for i in range(len(self.sorted_map)):
                sym, value = self.sorted_map[i]
                if sym == symbol:
                    self.sorted_map[i] = (symbol, v)
                    return
            self.sorted_map.append((symbol, v))
        else:
            self.hash_map[symbol] = v

    def __delitem__(self, __v) -> None:
        raise NotImplementedError

    def __getitem__(self, symbol: MatchableMixin):
        if isinstance(symbol, CompoundMatchableMixin):
            for sym, value in self.sorted_map:
                if sym == symbol:
                    return value
            if self.default_factory is not None:
                value = self.default_factory()
                self.sorted_map.append((symbol, value))
                return value
        return self.hash_map.__getitem__(symbol)

    def __len__(self) -> int:
        return len(self.sorted_map) + len(self.hash_map)

    def __iter__(self):
        yield from map(lambda tup: tup[0], self.sorted_map)
        yield from self.hash_map

    def items(self):
        yield from self.sorted_map
        yield from self.hash_map.items()

    def items_non_epsilon(self):
        for symbol, val in self.items():
            if not symbol.match(0, "ε"):
                yield symbol, val

    def clear(self) -> None:
        self.sorted_map = []
        if self.default_factory is None:
            self.hash_map = {}
        else:
            self.hash_map = defaultdict(self.default_factory)

    def update(self, m, **kwargs) -> None:
        for k, v in m.items():
            self[k] = v

    def match_atom(
        self, text: str, position: int, default
    ) -> tuple[MatchableMixin | str, set[State] | State]:
        for sym, value in self.sorted_map:
            if sym.match(position, text):
                return sym, value
        return text[position], self.hash_map.get(text[position], default)

    def __repr__(self):
        return repr(dict(self.items()))


class FiniteStateAutomaton(
    defaultdict[State, SymbolDispatchedMapping],
    ABC,
):
    states: set[State] | set[DFAState]
    start_state: State | DFAState
    accept: State | set[DFAState]
    symbols: set[MatchableMixin]

    @abstractmethod
    def transition(
        self, state: State, text: str, position: int
    ) -> DFAState | Collection[State]:
        pass

    @abstractmethod
    def transition_is_possible(
        self, state: State, text: str, position: int
    ) -> Optional[State]:
        pass

    @abstractmethod
    def all_transitions(self) -> tuple[MatchableMixin, State, State]:
        pass

    @abstractmethod
    def _dict(
        self,
    ) -> defaultdict[State, MutableMapping[MatchableMixin, DFAState | list[State]]]:
        pass

    def draw_with_graphviz(self):
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
                else:
                    dot.node(
                        f"{s1.id}",
                        shape="doublecircle" if s1.accepts else "circle",
                    )
                seen.add(s1)
            if s2 not in seen:
                seen.add(s2)
                dot.node(f"{s2.id}", shape="doublecircle" if s2.accepts else "circle")
            dot.edge(str(s1.id), str(s2.id), label=str(symbol))

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
