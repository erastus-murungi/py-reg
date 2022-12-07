from collections import defaultdict
from parser import Epsilon, RegexParser
from pprint import pprint

from core import TransitionsProvider
from dfa import DFA
from nfa import NFA
from simplify import simplify


class CompiledRegex(DFA):
    def __init__(self, nfa: NFA):
        super().__init__(nfa=nfa)
        self.minimize()


def compile_regex(regex: str) -> CompiledRegex:
    simplified_regex = simplify(regex)
    parser = RegexParser(simplified_regex)
    transitions = defaultdict(lambda: TransitionsProvider(set))
    start_state, final_state = parser.root.to_fsm(transitions)

    symbols = set()
    states = set()
    for state in transitions:
        states.add(state)
        for sym, end_state in transitions[state].items():
            states = states | end_state
            symbols.add(sym)
    symbols.discard(Epsilon)
    compiled_regex = CompiledRegex(
        NFA(transitions, states, symbols, start_state, final_state)
    )
    return compiled_regex


if __name__ == "__main__":
    # r = r"a*b+a.a*b|d+[A-Z]?"
    # compiled = compile_regex(r)
    # pprint(compiled)
    r = r"a*b+a.a*b|d+[A-Z]?"
    compiled = compile_regex(r)
    pprint(compiled)
