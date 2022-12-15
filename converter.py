from itertools import product
from typing import Iterable

from core import DFA, NFA, State, Transition


def _remove_unreachable_states(
    start_state: State,
    states: Iterable[State],
    transitions: NFA,
):
    # remove unreachable states
    seen = set()

    stack = [start_state]
    reachable = set()

    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)

        for _, v in transitions[u]:
            stack.extend(v)
        reachable.add(u)

    for state in states:
        if state not in reachable:
            transitions.pop(state)


def _compute_state_2_closure_mapping(nfa) -> dict[State, frozenset[State]]:
    state2closure = {}
    for state in nfa.states:
        state2closure[state] = nfa.epsilon_closure([state])
    return state2closure


def _mutate_and_gen_accept_states(
    states: Iterable[State], state2closure: dict[State, frozenset[State]]
) -> set[State]:
    accept_states = set()
    for state in states:
        if any(v.accepts for v in state2closure[state]):
            state.accepts = True
            accept_states.add(state)
    return accept_states


def _fill_transitions_and_gen_states(
    nfa: NFA,
    state2closure: dict[State, frozenset[State]],
    fsm: NFA,
) -> None:
    #  Construct transitions between `i` and `j` if there is some intermediary state k where
    # • there’s an ε-path i -> k
    # • there’s a non-ε transition k -> j

    for i, j in product(nfa.states, repeat=2):
        for k in state2closure[i]:
            for symbol in nfa.symbols:
                if j in nfa.transition(k, symbol):
                    fsm[i].add(Transition(symbol, j))
    fsm.update_symbols_and_states()


def remove_epsilon_transitions(nfa: NFA) -> DFA:
    state2closure = _compute_state_2_closure_mapping(nfa)
    dfa = DFA()
    _fill_transitions_and_gen_states(nfa, state2closure, dfa)
    _remove_unreachable_states(nfa.start, dfa.states, dfa)
    accept_states = _mutate_and_gen_accept_states(dfa.states, state2closure)
    dfa.accept.update(accept_states)
    dfa.set_start(nfa.start)
    return dfa
