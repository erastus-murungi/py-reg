from collections import defaultdict
from parser import Epsilon, RegexParser
from pprint import pprint

from core import RegexFlag, State, Transition
from dfa import DFA
from nfa import NFA
from simplify import simplify


class CompiledRegex(DFA):
    def __init__(self, regex: str):
        simplified_regex = simplify(regex)
        parser = RegexParser(simplified_regex)
        nfa = NFA()
        start_state, final_state = parser.root.fsm(nfa)
        nfa.update_symbols_and_states()
        nfa.symbols.discard(Epsilon)
        nfa.set_start(start_state)
        nfa.set_accept(final_state)
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


if __name__ == "__main__":
    # r = r"a*b+a.a*b|d+[A-Z]?"
    # compiled = compile_regex(r)
    # pprint(compiled)
    r = r"a*b+a.a*b|d+[A-Z]?"
    compiled = CompiledRegex(r)
    pprint(compiled)
