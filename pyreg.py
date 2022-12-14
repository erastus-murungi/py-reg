from collections import defaultdict
from parser import Epsilon, RegexParser
from pprint import pprint

from core import RegexFlag, State, Transition
from dfa import DFA
from nfa import NFA
from simplify import simplify


class CompiledRegex(DFA):
    def __init__(self, nfa: NFA):
        super().__init__(nfa=nfa)
        self.minimize()

    def match(
        self, state: State, text: str, position: int, flags: RegexFlag
    ) -> list[Transition]:
        return [
            transition
            for transition in self[state]
            if transition.match(text, position, flags)
        ]


def compile_regex(regex: str) -> CompiledRegex:
    simplified_regex = simplify(regex)
    parser = RegexParser(simplified_regex)
    transitions = defaultdict(set)
    start_state, final_state = parser.root.fsm(transitions)

    symbols = set()
    states = set()
    for state in transitions:
        states.add(state)
        for sym, end_state in transitions[state]:
            states.add(end_state)
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
