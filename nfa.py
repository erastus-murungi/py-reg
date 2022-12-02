import operator as op
from collections import defaultdict
from functools import reduce
from typing import Optional, Iterable

from core import State, Symbol, NullState, FiniteStateAutomaton, DFAState
from simplify import simplify
from utils import (
    BIN_OPERATORS,
    ALL_OPERATORS,
    EPSILON,
    precedence,
    gen_symbols_exclude_precedence_ops,
)

StatePair = tuple[State, State]


def format_regexp(regexp: str) -> str:
    """
    Insert . between atoms
    """
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


class NFA(FiniteStateAutomaton):
    """Formally, an NFA is a 5-tuple (Q, Σ, q0, T, δ) where
        • Q is finite set of states;
        • Σ is alphabet of input symbols;
        • q0 is start state;
        • T is subset of Q giving the ``accept`` states;
        and
        • δ is the transition function.
    Now the transition function specifies a set of states rather than a state: it maps Q × Σ to { subsets of Q }."""

    def _dict(self) -> defaultdict[State, defaultdict[Symbol, list[State]]]:
        d = defaultdict(lambda: defaultdict(list))
        for symbol, s1, s2 in self.all_transitions():
            d[s1][symbol].append(s2)
        return d

    def __init__(
        self,
        transitions: Optional[dict[State, dict[Symbol, list[State]]]] = None,
        states: Optional[set[State]] = None,
        symbols: Optional[set[Symbol]] = None,
        start_state: Optional[State] = None,
        accept: Optional[State] = None,
        *,
        regexp: Optional[str] = None,
    ):
        super(FiniteStateAutomaton, self).__init__(lambda: defaultdict(list))

        if regexp is not None:
            self.states: set[State] = set()
            self.symbols: set[Symbol] = set()
            self.init_from_regexp(regexp)
        else:
            assert transitions is not None
            self.update(transitions)
            assert symbols is not None
            self.symbols = symbols
            assert states is not None
            self.states = states
            assert start_state is not None
            self.start_state = start_state
            assert accept is not None
            self.accept = accept

    def subexpression(self, sub_expr, start_states, accept_states):
        frm, to = State.get_pair()
        self[frm][sub_expr].append(to)
        start_states.append(frm)
        accept_states.append(to)

    def init_from_regexp(self, regexp: str):
        postfix_regexp: str = shunting_yard(simplify(regexp))

        start_states: list[State] = []
        accept_states: list[State] = []
        symbols = gen_symbols_exclude_precedence_ops(postfix_regexp)

        for char in postfix_regexp:
            match char:
                case "|":
                    self.alternation(start_states, accept_states)
                case ".":
                    self.concatenate(start_states, accept_states)
                case "*":
                    self.zero_or_more(start_states, accept_states)
                case "+":
                    self.one_or_more(start_states, accept_states)
                case "?":
                    self.zero_or_one(start_states, accept_states)
                case _:
                    if char in symbols:
                        self.subexpression(char, start_states, accept_states)
                    else:
                        raise ValueError(f"{char} not understood")

        self.update_start_and_final_states(start_states, accept_states)
        self.update_states_set()
        self.symbols.update(symbols - {EPSILON})

    def update_start_and_final_states(
        self, start_states: list[State], accept_states: list[State]
    ):
        final_state: State = accept_states.pop()
        final_state.accepts = True
        assert not accept_states

        start_state: State = start_states.pop()
        start_state.is_start = True
        assert not start_states

        self.start_state = start_state
        self.accept = final_state

    def all_transitions(self):
        for state1, table in self.items():
            for symbol, state2s in table.items():
                for state2 in state2s:
                    yield symbol, state1, state2

    def transition(self, state: State, symbol: Symbol) -> list[State]:
        return self[state].get(symbol, [NullState])

    def states_eq(self, state_pair: StatePair) -> bool:
        state1, state2 = state_pair
        # both states should be accepting or both non_accepting
        if state1.accepts ^ state2.accepts:
            return False
        for symbol in self.symbols:
            if self.transition(state1, symbol) != self.transition(state2, symbol):
                return False
        return True

    # noinspection DuplicatedCode
    def alternation(
        self,
        start_states: list[State],
        accept_states: list[State],
    ) -> None:

        lower_start, upper_start = (
            start_states.pop(),
            start_states.pop(),
        )
        lower_accept, upper_accept = (
            accept_states.pop(),
            accept_states.pop(),
        )

        new_start, new_accept = State.get_pair()

        self[new_start][EPSILON].append(lower_start)
        self[new_start][EPSILON].append(upper_start)
        self[lower_accept][EPSILON].append(new_accept)
        self[upper_accept][EPSILON].append(new_accept)

        start_states.append(new_start)
        accept_states.append(new_accept)

    def concatenate(
        self,
        start_states: list[State],
        accept_states: list[State],
    ) -> None:
        self[accept_states.pop(-2)][EPSILON].append(start_states.pop())

    # noinspection DuplicatedCode
    def zero_or_more(
        self,
        start_states: list[State],
        accept_states: list[State],
    ) -> None:
        new_start, new_accept = State.get_pair()

        self[accept_states[-1]][EPSILON].append(start_states[-1])
        self[new_start][EPSILON].append(start_states.pop())
        self[accept_states.pop()][EPSILON].append(new_accept)
        self[new_start][EPSILON].append(new_accept)

        start_states.append(new_start)
        accept_states.append(new_accept)

    def one_or_more(self, start_states: list[State], accept_states):
        self[accept_states[-1]][EPSILON].append(start_states[-1])

    def zero_or_one(self, start_states: list[State], accept_states):
        self[start_states[-1]][EPSILON].append(accept_states[-1])

    def __repr__(self):
        return (
            f"FSM(states={self.states}, "
            f"symbols={self.symbols}, "
            f"start_state={self.start_state}, "
            f"accept_states={self.accept}) "
        )

    def epsilon_closure(self, states: Iterable):
        """
        This is the set of all the nodes which can be reached by following epsilon labeled edges
        This is done here using a depth first search

        https://castle.eiu.edu/~mathcs/mat4885/index/Webview/examples/epsilon-closure.pdf
        """

        seen = set()

        stack = list(states)
        closure = set()

        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)

            stack.extend(self[u][EPSILON])
            closure.add(u)

        return closure

    def move(self, states: Iterable, symbol: Symbol) -> frozenset[State]:
        return frozenset(reduce(op.add, (self[state][symbol] for state in states), []))

    def find_state(self, state_id: int) -> Optional[State]:
        for state in self.states:
            if state_id == state.id:
                return state
        return None

    def gen_dfa_state_set_flags(self, sources) -> DFAState:
        state = DFAState(from_states=sources)
        if self.start_state in state.sources:
            state.is_start = True
        if self.accept in state.sources:
            state.accepts = True
        return state

    def compute_transitions_for_dfa_state(
        self,
        dfa,
        dfa_from: DFAState,
        seen: set[frozenset],
        stack: list[DFAState],
    ):
        # what is the epsilon closure of the dfa_states
        eps = self.epsilon_closure(dfa_from.sources)
        d = self.gen_dfa_state_set_flags(eps)
        if d.accepts:
            dfa.accept.add(d)
        # next we want to see which states are reachable from each of the states in the epsilon closure
        for symbol in self.symbols:
            next_states_set = self.epsilon_closure(self.move(eps, symbol))
            # new DFAState
            df = self.gen_dfa_state_set_flags(next_states_set)
            dfa[d][symbol] = df
            if next_states_set not in seen:
                seen.add(next_states_set)
                stack.append(df)
        return d


if __name__ == "__main__":
    nf = NFA(regexp="ab")
    nf.draw_with_graphviz()
    print(nf)
