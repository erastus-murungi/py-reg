import operator as op
from collections import defaultdict
from functools import reduce
from itertools import combinations
from typing import Optional, Iterable, Callable

import graphviz

from core import State, Symbol, EMPTY_STATE, DFAState
from simplify import simplify_extensions
from utils import (
    is_iterable,
    BIN_OPERATORS,
    ALL_OPERATORS,
    EPSILON,
    precedence,
    PRECEDENCE,
    UNION,
    CONCATENATION,
    KLEENE_CLOSURE,
)


def format_regexp(regexp: str) -> str:
    if len(regexp) == 0:
        return EPSILON
    fmt_regexp = []
    for c1, c2 in zip(regexp, regexp[1:]):
        fmt_regexp.append(c1)
        if (
            (c1 != "(" and c2 != ")")
            and c2 not in ALL_OPERATORS
            and c1 not in BIN_OPERATORS
        ):
            fmt_regexp.append(".")
    fmt_regexp.append(regexp[-1])
    return "".join(fmt_regexp)


def shunting_yard(infix: str) -> str:
    stack = []
    postfix = []
    fmt_infix = format_regexp(infix)
    for c in fmt_infix:
        if c == "(":
            stack.append(c)
        elif c == ")":
            while stack[-1] != "(":
                postfix.append(stack.pop())
            stack.pop()
        else:
            while stack:
                peeked = stack[-1]
                peeked_precedence = precedence(peeked)
                current_precedence = precedence(c)
                if peeked_precedence >= current_precedence:
                    postfix.append(stack.pop())
                else:
                    break
            stack.append(c)
    while stack:
        postfix.append(stack.pop())
    return "".join(postfix)


class NFA:
    """Formally, an NFA is a 5-tuple (Q, Σ, q0, T, δ) where
        • Q is finite set of states;
        • Σ is alphabet of input symbols;
        • q0 is start state;
        • T is subset of Q giving the ``accept`` states;
        and
        • δ is the transition function.
    Now the transition function specifies a set of states rather than a state: it maps Q × Σ to { subsets of Q }."""

    save_final: Optional[State]

    def __init__(
        self,
        states: set[State],
        symbols: set[Symbol],
        transition_table: dict[State, dict[Symbol, Iterable]]
        | Callable[[State, Symbol], State]
        | dict[State, dict[Symbol, State]]
        | Callable[[State, Symbol], Iterable],
        start_state: State,
        accepting_states: set[State] | frozenset[State],
    ):
        self.states = states
        self.symbols = symbols
        self.transition_table = transition_table
        self.start_state = start_state
        self.accept_states = accepting_states

    @staticmethod
    def from_regexp(regexp: str) -> "NFA":
        postfix_regexp: str = shunting_yard(simplify_extensions(regexp))
        states, symbols, start_state, final_state, transition_table = NFA.regexp_to_nfa(
            postfix_regexp
        )
        return NFA(states, symbols, transition_table, start_state, {final_state})

    @staticmethod
    def get_transition_function(transition_table) -> Callable[[State, Symbol], State]:
        def _transition_function(state, symbol):
            assert isinstance(state, State)
            assert isinstance(symbol, Symbol)
            try:
                return transition_table[state][symbol]
            except KeyError:
                return EMPTY_STATE

        return _transition_function

    @property
    def transitions(self):
        for s1, table in self.transition_table.items():
            for symbol, s2s in table.items():
                if is_iterable(s2s):
                    for s2 in s2s:
                        yield (symbol, s1, s2)
                else:
                    yield (symbol, s1, s2s)

    def make_transition(self, state, symbol):
        try:
            return self.transition_table[state][symbol]
        except KeyError:
            return EMPTY_STATE

    def states_eq(self, state1: State, state2: State):
        # both states should be accepting or both non_accepting
        if state1.accepts ^ state2.accepts:
            return False
        for symbol in self.symbols:
            if self.make_transition(state1, symbol) != self.make_transition(
                state2, symbol
            ):
                return False
        return True

    @staticmethod
    def get_states_pair():
        return State(), State()

    # noinspection DuplicatedCode
    @staticmethod
    def union(
        lower_start: State,
        upper_start: State,
        lower_accept: State,
        upper_accept: State,
        start_states_stack: list[State],
        accept_states_stack: list[State],
        transition_table: dict[State, dict[Symbol, list[State]]],
    ) -> None:
        new_start, new_accept = NFA.get_states_pair()

        transition_table[new_start][EPSILON].append(lower_start)
        transition_table[new_start][EPSILON].append(upper_start)
        transition_table[lower_accept][EPSILON].append(new_accept)
        transition_table[upper_accept][EPSILON].append(new_accept)

        start_states_stack.append(new_start)
        accept_states_stack.append(new_accept)

    @staticmethod
    def concatenate(
        start_state: State,
        accept_state: State,
        accept_states_stack: list[State],
        transition_table: dict[State, dict[Symbol, list[State]]],
    ) -> None:
        transition_table[start_state][EPSILON].append(accept_state)
        accept_states_stack.append(NFA.save_final)
        NFA.save_final = None

    # noinspection DuplicatedCode
    @staticmethod
    def kleene(
        start_state: State,
        accept_state: State,
        start_states_stack: list[State],
        accept_states_stack: list[State],
        transition_table: dict[State, dict[Symbol, list[State]]],
    ) -> None:
        new_start, new_accept = NFA.get_states_pair()

        transition_table[accept_state][EPSILON].append(start_state)
        transition_table[new_start][EPSILON].append(new_accept)
        transition_table[new_start][EPSILON].append(start_state)
        transition_table[accept_state][EPSILON].append(new_accept)

        start_states_stack.append(new_start)
        accept_states_stack.append(new_accept)

    @staticmethod
    def compute_symbol_set(postfix_regexp) -> set[Symbol]:
        return set(postfix_regexp) - set(PRECEDENCE.keys())

    @staticmethod
    def regexp_to_nfa(
        postfix_regexp: str,
    ) -> tuple[
        set[State],
        set[Symbol],
        State,
        State,
        dict[State, dict[Symbol, list[State]]],
    ]:
        start_states_stack: list[State] = []
        accept_states_stack: list[State] = []
        transition_table: dict[State, dict[Symbol, list[State]]] = defaultdict(
            lambda: defaultdict(list)
        )
        symbols: set[Symbol] = NFA.compute_symbol_set(postfix_regexp)
        final_state: State = EMPTY_STATE
        for i, c in enumerate(postfix_regexp):
            if c in symbols:
                frm, to = NFA.get_states_pair()
                transition_table[frm][c].append(to)
                start_states_stack.append(frm)
                accept_states_stack.append(to)

            elif c == UNION:
                lower_start, upper_start = (
                    start_states_stack.pop(),
                    start_states_stack.pop(),
                )
                lower_accept, upper_accept = (
                    accept_states_stack.pop(),
                    accept_states_stack.pop(),
                )
                NFA.union(
                    lower_start,
                    upper_start,
                    lower_accept,
                    upper_accept,
                    start_states_stack,
                    accept_states_stack,
                    transition_table,
                )

            elif c == CONCATENATION:
                NFA.save_final = accept_states_stack.pop()
                NFA.concatenate(
                    accept_states_stack.pop(),
                    start_states_stack.pop(),
                    accept_states_stack,
                    transition_table,
                )

            elif c == KLEENE_CLOSURE:
                NFA.kleene(
                    start_states_stack.pop(),
                    accept_states_stack.pop(),
                    start_states_stack,
                    accept_states_stack,
                    transition_table,
                )
            if i == (len(postfix_regexp) - 1):
                final_state: State = accept_states_stack.pop()
                final_state.accepts = True
                if EPSILON in symbols:
                    symbols.remove(EPSILON)

        initial_state = NFA.get_initial_state(start_states_stack)
        states = NFA.compute_states_set(transition_table)
        return states, symbols, initial_state, final_state, transition_table

    @staticmethod
    def compute_states_set(transition_table) -> set[State]:
        states = set()
        for s1, table in transition_table.items():
            states.add(s1)
            for s2s in table.values():
                if is_iterable(s2s):
                    for s2 in s2s:
                        states.add(s2)
                else:
                    states.add(s2s)
        return states

    def __repr__(self):
        return (
            f"FSM(states={self.states}, "
            f"symbols={self.symbols}, "
            f"start_state={self.start_state}, "
            f"accept_states={self.accept_states}) "
        )

    @staticmethod
    def get_initial_state(start_states_stack: list[State]):
        # after processing, start_states_stack should only have one item
        assert len(start_states_stack) == 1
        start_state: State = start_states_stack.pop()
        start_state.is_start = True
        return start_state

    def draw_with_graphviz(self):
        dot = graphviz.Digraph(
            self.__class__.__name__ + "WHP",
            format="pdf",
            engine="circo",
        )
        dot.attr("node", shape="circle")

        for symbol, s1, s2 in self.transitions:
            if s1.is_start:
                dot.node(
                    str(s1.id),
                    color="green",
                    shape="doublecircle" if s1.accepts else "circle",
                )
                dot.node("start", shape="none")
                dot.edge("start", f"{s1.id}", arrowhead="vee")
            else:
                dot.node(
                    f"{s1.id}",
                    shape="doublecircle" if s1.accepts else "circle",
                )

            dot.node(f"{s2.id}", shape="doublecircle" if s2.accepts else "circle")
            dot.edge(str(s1.id), str(s2.id), label=symbol)

        dot.render(view=True)

    def _one_epsilon_closure_helper(self, s0: State, seen: set):
        seen.add(s0)
        closure = self.transition_table[s0][EPSILON][:]  # need to do a copy
        subs = []
        for s in closure:
            if (
                s not in seen
            ):  # to prevent infinite recursion when we encounter cycles in the NFA
                subs.append(self._one_epsilon_closure_helper(s, seen))
        return [s0] + closure + reduce(op.add, subs, [])

    def _one_epsilon_closure(self, s0: State) -> list[State]:
        return self._one_epsilon_closure_helper(s0, set())

    def epsilon_closure(self, states: Iterable):
        return frozenset(
            reduce(op.add, (self._one_epsilon_closure(state) for state in states), [])
        )

    def move(self, states: Iterable, symbol: Symbol) -> frozenset[State]:
        return frozenset(
            reduce(
                op.add, (self.transition_table[state][symbol] for state in states), []
            )
        )

    def find_state(self, state_id: int) -> Optional[State]:
        for state in self.states:
            if state_id == state.id:
                return state
        return None

    def get_dfa_state(self, from_states):
        dfa_from = DFAState(from_states=from_states)
        if self.start_state in dfa_from.from_states:
            dfa_from.is_start = True
        for accept_state in self.accept_states:
            if accept_state in dfa_from.from_states:
                dfa_from.accepts = True
                break
        return dfa_from

    def compute_transitions_for_dfa_state(
        self,
        dfa_from: DFAState,
        transition_table: dict[DFAState, dict[Symbol, DFAState]],
        seen: set[frozenset],
        stack: list[DFAState],
        accept_states: set[State],
    ):
        # what is the epsilon closure of the dfa_states
        eps = self.epsilon_closure(dfa_from.from_states)
        d = self.get_dfa_state(eps)
        if d.accepts:
            accept_states.add(d)
        # next we want to see which states are reachable from each of the states in the epsilon closure
        for symbol in self.symbols:
            next_states_set = self.epsilon_closure(self.move(eps, symbol))
            # new DFAState
            df = self.get_dfa_state(next_states_set)
            transition_table[d][symbol] = df
            if next_states_set not in seen:
                seen.add(next_states_set)
                stack.append(df)
        return d

    @staticmethod
    def clean_up_empty_sets(transition_table: dict[DFAState, dict[Symbol, DFAState]]):
        pruned_transition_table = defaultdict(dict)
        for start_state, table in transition_table.items():
            for symbol, end_state in table.items():
                if end_state.from_states:
                    pruned_transition_table[start_state][symbol] = end_state
        return pruned_transition_table

    @staticmethod
    def find_indistinguishable_pairs(states: set[State]) -> list[tuple[State, State]]:
        state_pairs = combinations(states, 2)
        return list(filter(lambda pair: NFA.states_eq(*pair), state_pairs))

    def to_dfa(self):
        pass
