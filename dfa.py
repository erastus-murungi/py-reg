from collections import defaultdict
from itertools import combinations
from typing import Iterator, Optional

from more_itertools import minmax

from core import (DFAState, FiniteStateAutomaton, MatchableMixin, NullDfaState,
                  State, SymbolDispatchedMapping)
from data_structures import UnionFind
from nfa import NFA


class DFA(FiniteStateAutomaton):
    def __init__(
        self,
        transitions: Optional[dict[DFAState, SymbolDispatchedMapping]] = None,
        states: Optional[set[DFAState]] = None,
        symbols: Optional[set[MatchableMixin]] = None,
        start_state: Optional[DFAState] = None,
        accept: Optional[set[DFAState]] = None,
        *,
        nfa: Optional[NFA] = None
    ):
        super(FiniteStateAutomaton, self).__init__(SymbolDispatchedMapping)

        if nfa is not None:
            self.states: set[DFAState] = set()
            self.symbols: set[MatchableMixin] = set()
            self.accept: set[DFAState] = set()
            self.subset_construction(nfa)
        else:
            assert transitions is not None
            self.update(transitions)
            assert symbols is not None
            self.symbols = symbols
            assert states is not None
            self.states = states
            assert start_state is not None
            self.set_start(start_state)
            assert accept is not None
            self.accept = accept

    def transition(self, state: State, symbol: MatchableMixin) -> State:
        return self[state].get(symbol, NullDfaState)

    def subset_construction(self, nfa: NFA):
        s0 = DFAState(from_states=frozenset({nfa.start_state}))
        seen, stack = set(), []
        self.set_start(nfa.compute_transitions_for_dfa_state(self, s0, seen, stack))

        while stack:
            nfa.compute_transitions_for_dfa_state(self, stack.pop(), seen, stack)

        self.clean_up_empty_sets()
        self.update_states_set()
        self.symbols = nfa.symbols

    def clean_up_empty_sets(self):
        items = self._dict().items()
        self.clear()
        for start_state, table in items:
            for symbol, end_state in table.items():
                if end_state.sources:
                    self[start_state][symbol] = end_state

    def all_transitions(self):
        for state1, table in self.items():
            for symbol, state2 in table.items():
                yield symbol, state1, state2

    def _dict(self) -> defaultdict[DFAState, dict[MatchableMixin, DFAState]]:
        d = defaultdict(dict)
        for symbol, s1, s2 in self.all_transitions():
            d[s1][symbol] = s2
        return d

    def gen_equivalence_states(self) -> Iterator[set[State]]:
        """
        Myhill-Nerode Theorem
        https://www.cs.scranton.edu/~mccloske/courses/cmps364/dfa_minimize.html
        """

        # a state is indistinguishable from itself
        indistinguishable = {(p, p) for p in self.states}

        for p, q in combinations(self.states, 2):
            # a pair of states are maybe indistinguishable
            # if they are both accepting or both non-accepting
            # we use min max to provide an ordering based on the labels
            p, q = minmax(p, q)
            if p.accepts == q.accepts:
                indistinguishable.add((p, q))

        union_find = UnionFind(self.states)

        changed = True
        while changed:
            changed = False
            removed = set()
            for p, q in indistinguishable:
                if p == q:
                    continue
                # if two states are maybe indistinguishable, then do some more work to prove they are actually
                # indistinguishable
                for a in self.symbols:
                    km = minmax(self.transition(p, a), self.transition(q, a))
                    if (
                        km != (NullDfaState, NullDfaState)
                        and km not in indistinguishable
                    ):
                        removed.add((p, q))
                        changed = True
            indistinguishable = indistinguishable - removed

        for p, q in indistinguishable:
            union_find.union(p, q)

        return union_find.to_sets()

    def gen_dfa_state_set_flags(self, sources):
        if len(sources) == 1:
            return sources.pop()
        state = DFAState(from_states=frozenset(sources))
        if self.start_state in state.sources:
            state.is_start = True
        for accept_state in self.accept:
            if accept_state in state.sources:
                state.accepts = True
                break
        return state

    def minimize(self):
        self.states: set[DFAState] = set(
            map(self.gen_dfa_state_set_flags, self.gen_equivalence_states())
        )
        lost = {
            original: compound
            for compound in self.states
            for original in compound.sources
            if len(compound.sources) > 1
        }

        for a in list(self.keys()):
            if a in lost:
                self[lost.get(a)] = self.pop(a)
        for a in self:
            for symbol, b in self[a].items():
                if b in lost:
                    self[a][symbol] = lost.get(b)

        (self.start_state,) = tuple(filter(lambda s: s.is_start, self.states))
        self.accept = set(filter(lambda s: s.accepts, self.accept))

    def transition_is_possible(
        self, state: State, text: str, position: int
    ) -> Optional[State]:
        _, val = self[state].match_atom(text, position, None)
        return val


if __name__ == "__main__":
    # df = DFA(regexp=r"(ab?)c*")
    # df.draw_with_graphviz()
    # df.minimize()
    # df.draw_with_graphviz()

    # A = DFAState(frozenset(range(1)), is_start=True)
    # B = DFAState(frozenset(range(2)))
    # C = DFAState(frozenset(range(3)), accepts=True)
    # D = DFAState(
    #     frozenset(range(4)),
    # )
    # E = DFAState(frozenset(range(5)), accepts=True)
    #
    # transitions = {
    #     A: {"a": B, "b": D},
    #     B: {"a": C, "b": E},
    #     C: {"a": B, "b": E},
    #     D: {"a": C, "b": E},
    #     E: {"a": E, "b": E},
    # }
    #
    # dfa1 = DFA(transitions, {A, B, C, D, E}, {"a", "b"}, A, {C, E})
    # dfa1.draw_with_graphviz()
    # dfa1.minimize()
    # dfa1.draw_with_graphviz()

    # q0 = DFAState(frozenset([0]), is_start=True)
    # q1 = DFAState(frozenset([1]))
    # q2 = DFAState(frozenset([2]))
    # q3 = DFAState(frozenset([3]))
    # q4 = DFAState(frozenset([4]), accepts=True)
    #
    # a = Character("a")
    # b = Character("b")
    # transitions = {
    #     q0: {a: q1, b: q2},
    #     q1: {a: q1, b: q3},
    #     q2: {b: q2, a: q1},
    #     q3: {a: q1, b: q4},
    #     q4: {a: q1, b: q2},
    # }
    #
    # dfa3 = DFA(
    #     transitions, {q0, q1, q2, q3, q4}, {Character("a"), Character("b")}, q0, {q4}
    # )
    # dfa3.draw_with_graphviz()
    #
    # dfa3.minimize()
    # dfa3.draw_with_graphviz()

    ...
