from itertools import product
from typing import Iterable

from src.core import NFA, State, Tag, Transition, gen_state


def _remove_unreachable_states(
    start_state: State,
    states: Iterable[State],
    nfa: NFA,
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

        stack.extend([transition.end for transition in nfa[u]])
        reachable.add(u)

    for state in states:
        if state not in reachable:
            nfa.pop(state)


def _compute_state_2_closure_mapping(nfa) -> dict[State, frozenset[State]]:
    state2closure = {}
    for state in nfa.states:
        state2closure[state] = nfa.epsilon_closure([state])
    return state2closure


def _mutate_and_gen_accept_states(
    old_nfa: NFA,
    new_nfa: NFA,
    state2closure: dict[State, frozenset[State]],
):
    final_accept = gen_state()
    added_transitions = {}
    for state in new_nfa.states:
        if any(v in old_nfa.accepting_states for v in state2closure[state]):
            transition = Transition(Tag.epsilon(), final_accept)
            added_transitions[state] = transition
    if len(added_transitions) == 0:
        raise RuntimeError(f"No accept states found in {old_nfa}")
    if len(added_transitions) == 1:
        new_nfa.accepting_states.update(old_nfa.accepting_states)
    else:
        for state, transition in added_transitions.items():
            new_nfa[state].append(transition)
        new_nfa.accepting_states.add(final_accept)


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
                    fsm.add_transition(i, j, symbol)
    fsm.update_symbols_and_states()


def reduce_epsilon_transitions(nfa: NFA) -> NFA:
    state2closure = _compute_state_2_closure_mapping(nfa)
    new_nfa = NFA()
    _fill_transitions_and_gen_states(nfa, state2closure, new_nfa)
    _remove_unreachable_states(nfa.start_state, new_nfa.states, new_nfa)
    _mutate_and_gen_accept_states(nfa, new_nfa, state2closure)
    new_nfa.set_start(nfa.start_state)
    return new_nfa
