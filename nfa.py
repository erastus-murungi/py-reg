from collections import defaultdict
from functools import reduce
from itertools import combinations_with_replacement, product
from symbol import (Epsilon, AllOps, BinOps, ClosingParen, Concatenate,
                    OpeningParen, Operator, Symbol,
                    gen_symbols_exclude_precedence_ops, precedence)
from typing import Iterable, Optional

from core import DFAState, FiniteStateAutomaton, NullState, State
from data_structures import SymbolDispatchedMapping
from simplify import simplify

StatePair = tuple[State, State]


def splice_concatenate_operator(regexp: list[Symbol]) -> list[Symbol]:
    """
    Insert . between atoms
    """
    if not regexp:
        return Epsilon
    fmt_regexp = []
    for c1, c2 in zip(regexp, regexp[1:]):
        fmt_regexp.append(c1)
        if (
            (not c1.match(OpeningParen) and not c2.match(ClosingParen))
            and c2 not in AllOps - {OpeningParen}
            and c1 not in BinOps
        ):
            fmt_regexp.append(Concatenate)
    fmt_regexp.append(regexp[-1])
    return fmt_regexp


def shunting_yard(infix: list[Symbol]) -> list[Symbol]:
    stack, postfix = [], []
    fmt_infix = splice_concatenate_operator(infix)
    for c in fmt_infix:
        if c.match(OpeningParen):
            stack.append(c)
        elif c.match(ClosingParen):
            while not stack[-1].match(OpeningParen):
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
    return postfix


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
        d: defaultdict[State, defaultdict[Symbol, list[State]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for symbol, s1, s2 in self.all_transitions():
            d[s1][symbol].append(s2)
        return d

    def __init__(
        self,
        transitions: Optional[defaultdict[State, SymbolDispatchedMapping]] = None,
        states: Optional[set[State]] = None,
        symbols: Optional[set[Symbol]] = None,
        start_state: Optional[State] = None,
        accept: Optional[State] = None,
        *,
        regexp: Optional[str] = None,
    ):
        super(FiniteStateAutomaton, self).__init__(lambda: SymbolDispatchedMapping(set))
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
        frm, to = State.pair()
        self[frm][sub_expr].add(to)
        start_states.append(frm)
        accept_states.append(to)

    def init_from_regexp(self, regexp: str):
        postfix_regexp: list[Symbol] = shunting_yard(simplify(regexp))

        start_states: list[State] = []
        accept_states: list[State] = []
        symbols = gen_symbols_exclude_precedence_ops(postfix_regexp)

        for char in postfix_regexp:
            if isinstance(char, Operator):
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
                        raise ValueError(f"{char} not understood")
            else:
                if char in symbols:
                    self.subexpression(char, start_states, accept_states)
                else:
                    raise ValueError(f"{char} not understood")

        self.update_start_and_final_states(start_states, accept_states)
        self.update_states_set()
        self.symbols.update(symbols - {Epsilon})

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
        return self[state].match_atom(symbol, [NullState])

    def transition_is_possible(self, state: State, symbol: Symbol) -> bool:
        return self[state].match_atom(symbol, None)

    def states_eq(self, state_pair: StatePair) -> bool:
        state1, state2 = state_pair
        # both states should be accepting or both non_accepting
        if state1.accepts ^ state2.accepts:
            return False
        for symbol in self.symbols:
            if self.transition(state1, symbol) != self.transition(state2, symbol):
                return False
        return True

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

        new_start, new_accept = State.pair()

        self[new_start][Epsilon].add(lower_start)
        self[new_start][Epsilon].add(upper_start)
        self[lower_accept][Epsilon].add(new_accept)
        self[upper_accept][Epsilon].add(new_accept)

        start_states.append(new_start)
        accept_states.append(new_accept)

    def concatenate(
        self,
        start_states: list[State],
        accept_states: list[State],
    ) -> None:
        self[accept_states.pop(-2)][Epsilon].add(start_states.pop())

    # noinspection DuplicatedCode
    def zero_or_more(
        self,
        start_states: list[State],
        accept_states: list[State],
    ) -> None:
        new_start, new_accept = State.pair()

        self[accept_states[-1]][Epsilon].add(start_states[-1])
        self[new_start][Epsilon].add(start_states.pop())
        self[accept_states.pop()][Epsilon].add(new_accept)
        self[new_start][Epsilon].add(new_accept)

        start_states.append(new_start)
        accept_states.append(new_accept)

    def one_or_more(self, start_states: list[State], accept_states):
        self[accept_states[-1]][Epsilon].add(start_states[-1])

    def zero_or_one(self, start_states: list[State], accept_states):
        self[start_states[-1]][Epsilon].add(accept_states[-1])

    def __repr__(self):
        return (
            f"FSM(states={self.states}, "
            f"symbols={self.symbols}, "
            f"start_state={self.start_state}, "
            f"accept_states={self.accept}) "
        )

    def epsilon_closure(self, states: Iterable) -> frozenset:
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

            stack.extend(self[u][Epsilon])
            closure.add(u)

        return frozenset(closure)

    def move(self, states: Iterable, symbol: Symbol) -> frozenset[State]:
        return frozenset(
            reduce(set.union, (self[state][symbol] for state in states), set())
        )

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

    def epsilon_frontier(self, state: State, visited: set[State]):
        if self.transition_is_possible(state, Epsilon):
            for eps_state in self.transition(state, Epsilon):
                if eps_state not in visited:
                    visited.add(eps_state)
                    yield from self.epsilon_frontier(eps_state, visited)
        else:
            yield state

    def remove_epsilon_transitions(self):
        state2closure = {}

        for state in self.states:
            state2closure[state] = self.epsilon_closure([state])

        # new automaton transitions
        transitions = defaultdict(lambda: SymbolDispatchedMapping(set))

        #  Construct transitions between `i` and `j` if there is some intermediary state k where
        # • there’s an ε-path i -> k
        # • there’s a non-ε transition k -> j

        states = set()
        for i, j in product(self.states, repeat=2):
            # so we do this
            for k in state2closure[i]:
                for symbol in self.symbols:
                    if j in self.transition(k, symbol):
                        transitions[i][symbol].add(j)
                        states = states | {i, j}

        accept_states = set()
        for state in states:
            if any(v.accepts for v in state2closure[state]):
                state.accepts = True
                accept_states.add(state)

        # remove unreachable states
        seen = set()

        stack = [self.start_state]
        reachable = set()

        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)

            for _, v in transitions[u].items():
                stack.extend(v)
            reachable.add(u)

        for state in states:
            if state not in reachable:
                transitions.pop(state)

        return NFA(transitions, states, self.symbols, self.start_state, accept_states)


if __name__ == "__main__":
    nf = NFA(regexp=r"a*b+aa*")
