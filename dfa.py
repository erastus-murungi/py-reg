from collections import defaultdict
from itertools import combinations
from typing import Callable

from core import State, Symbol, DFAState
from nfa import NFA


def subset_construction(nfa: NFA):
    transitions_table: dict[DFAState, dict[Symbol, DFAState]] = defaultdict(dict)
    s0 = DFAState(from_states=frozenset({nfa.start_state}))
    accept_states: set[State] = set()
    seen = set()
    stack = [s0]

    to_explore = stack.pop()
    initial_state = nfa.compute_transitions_for_dfa_state(
        to_explore, transitions_table, seen, stack, accept_states
    )
    initial_state.is_start = True

    while stack:
        to_explore = stack.pop()
        nfa.compute_transitions_for_dfa_state(
            to_explore, transitions_table, seen, stack, accept_states
        )
    transitions_table = nfa.clean_up_empty_sets(transitions_table)
    states = NFA.compute_states_set(transitions_table)
    return states, nfa.symbols, transitions_table, initial_state, accept_states


class DFA(NFA):
    def __init__(
        self,
        states: set[State],
        symbols: set[Symbol],
        transition_table: Callable[[State, Symbol], State]
        | dict[State, dict[Symbol, State]],
        start_state: State,
        accepting_states: frozenset[State],
    ):
        super().__init__(
            states, symbols, transition_table, start_state, accepting_states
        )
        self.states = states
        self.symbols = symbols
        self.transition_table = transition_table
        self.start_state = start_state
        self.accept_states = accepting_states

    @staticmethod
    def from_regexp(regexp: str) -> "NFA":
        return DFA(*subset_construction(NFA.from_regexp(regexp)))

    @staticmethod
    def from_nfa(nfa):
        return DFA(*subset_construction(nfa))

    @staticmethod
    def distinguish(
        states: tuple[tuple[State, State]],
        distinguish_function: Callable[[State, State], bool],
    ) -> dict[tuple[State, State], bool]:
        return {
            states_pair: distinguish_function(*states_pair) for states_pair in states
        }

    @staticmethod
    def collapse(ds: list[frozenset[State]]) -> set[State]:
        collapsed = set()
        for d in ds:
            for state in d:
                collapsed.add(state)
        return collapsed

    @staticmethod
    def reversed_transition_table(
        transition_table: dict[State, dict[Symbol, State]]
    ) -> dict[State, dict[Symbol, list[State]]]:
        ret = defaultdict(lambda: defaultdict(list))
        for start_state, table in transition_table.items():
            for symbol, end_state in table.items():
                ret[end_state][symbol].append(start_state)
        return ret

    @staticmethod
    def equivalence_partition(iterable, relation):
        """Partitions a set of objects into equivalence classes

        Args:
            iterable: collection of objects to be partitioned
            relation: equivalence relation. I.e. relation(o1, o2) evaluates to True
                if and only if o1 and o2 are equivalent

        Returns: classes, partitions
            classes: A sequence of sets. Each one is an equivalence class
            partitions: A dictionary mapping objects to equivalence classes
        """

        classes = []
        for o in iterable:  # for each object
            # find the class it is in
            found = False
            for c in classes:
                if relation(next(iter(c)), o):  # is it equivalent to this class?
                    c.add(o)
                    found = True
                    break
            if not found:  # it is in a new class
                classes.append({o})
        return set(map(frozenset, classes))

    def populate_in_dist(self) -> dict[State, dict[State, bool]]:
        in_dist = defaultdict(dict)
        for p, q in combinations(self.states, 2):
            in_dist[min(p, q)][max(p, q)] = p.accepts ^ q.accepts
        for p in self.states:
            in_dist[p][p] = False

        changed = True
        while changed:
            changed = False
            for _p, _q in combinations(self.states, 2):
                p, q = min(_p, _q), max(_p, _q)
                if not in_dist[p][q]:
                    for a in self.symbols:
                        k, m = self.make_transition(p, a), self.make_transition(q, a)
                        if k in in_dist and m in in_dist:
                            if in_dist[min(k, m)][max(k, m)]:
                                in_dist[p][q] = True  # distinguishable
                                changed = True

        return in_dist

    def get_new_states(self, in_dist: dict[State, dict[State, bool]]) -> list[DFAState]:
        def relation(p, q):
            return not in_dist[min(p, q)][max(p, q)]

        equivalence_classes = self.equivalence_partition(self.states, relation)

        new_states: list[DFAState] = list(
            map(
                self.get_dfa_state,
                filter(
                    lambda equivalence_class: len(equivalence_class) > 1,
                    equivalence_classes,
                ),
            )
        )
        return new_states

    def minimize(self):
        # if in_dist[p][q] == True, (p, q) are distinguishable
        # https://www.cs.scranton.edu/~mccloske/courses/cmps364/dfa_minimize.html
        in_dist = self.populate_in_dist()
        new_states = self.get_new_states(in_dist)
        rev_trans_table = self.reversed_transition_table(self.transition_table)
        new_trans = self.transition_table.copy()

        for new_state in new_states:
            for from_state in new_state.from_states:
                for symbol, source_states in rev_trans_table[from_state].items():
                    for source_state in source_states:
                        new_trans[source_state][symbol] = new_state

            new_states_iterable = iter(new_state.from_states)
            v = next(new_states_iterable)
            new_trans[new_state] = new_trans[v]
            new_trans.pop(v)

            for v in new_states_iterable:
                new_trans.pop(v)

        (starting_state,) = tuple(filter(lambda s: s.is_start, new_trans))
        accepting_states = frozenset(filter(lambda s: s.accepts, new_trans))

        return DFA(
            set(new_trans.keys()),
            self.symbols.copy(),
            new_trans,
            starting_state,
            accepting_states,
        )
