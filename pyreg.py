from pprint import pprint

from core import DFA, NFA, RegexFlag, RegexParser, State, Transition
from simplify import simplify


class CompiledRegex(DFA):
    def __init__(self, regex: str):
        simplified_regex = simplify(regex)
        parser = RegexParser(simplified_regex)
        nfa = NFA()
        fragment = parser.root.fsm(nfa)
        nfa.update_symbols_and_states()
        nfa.set_start(fragment.start)
        nfa.set_accept(fragment.end)
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
