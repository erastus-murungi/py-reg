import re
from dataclasses import dataclass

from core import State, Symbol
from nfa import NFA
from utils import EPSILON


class Cursor:
    def __init__(self, start, end=None):
        self.start = start
        self.end = end or start

    def to_match(self, text):
        return Match(self.start, text[self.start : self.end])

    def increment(self):
        return Cursor(self.start, self.end + 1)

    def __repr__(self):
        return f"[{self.start}, {self.end})"


@dataclass(frozen=True)
class Match:
    start: int
    substr: str

    @property
    def span(self):
        return self.start, self.end()

    def end(self):
        return self.start + len(self.substr)


class Matcher:
    def __init__(self, pattern: str, regexp: str):
        self.text = pattern
        self.regexp = regexp
        self.nfa = NFA.from_regexp(regexp)

    def __iter__(self):
        """times specifies the order in which the dfs was finished"""
        start = 0

        while True:

            def explore_node(cursor: Cursor, state: State, symbol: Symbol):
                nodes = self.nfa.make_transition(state, symbol)
                for node in nodes:
                    if not node.is_empty():
                        yield from match(
                            cursor if symbol == EPSILON else cursor.increment(), node
                        )

            def match(cursor: Cursor, state: State):
                if state.accepts:
                    yield cursor
                if cursor.end < len(self.text):
                    yield from explore_node(cursor, state, self.text[cursor.end])
                yield from explore_node(cursor, state, EPSILON)

            new_start = start
            for c in match(Cursor(start), self.nfa.start_state):
                new_start = max(c.end, new_start)
                yield c.to_match(self.text)

            if start == new_start:
                break
            start = new_start

    def __repr__(self):
        return f"{self.__class__.__name__}(regex={self.regexp!r}, text={self.text!r})"


if __name__ == "__main__":
    regex = "(ab*)*"
    t = "abaabb"  # nfa.draw_with_graphviz()
    matcher = Matcher(t, regex)

    print(matcher)
    for span in matcher:
        print(span)

    for span in re.finditer(regex, t):
        print(span)
