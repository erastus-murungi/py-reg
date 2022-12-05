import re
from dataclasses import dataclass
from symbol import Epsilon, Character

from core import State
from dfa import DFA


@dataclass(frozen=True)
class Match:
    start: int
    end: int
    substr: str

    @property
    def span(self):
        return self.start, self.end


class Matcher:
    def __init__(self, pattern: str, regexp: str):
        self.text = pattern
        self.regexp = regexp
        self.dfa = DFA(regexp=regexp)
        self.dfa.minimize()

    def __iter__(self):
        def add_next_states(
            state: State, next_states: list[State], visited: set[State]
        ):
            if (
                eps_state := self.dfa.transition_is_possible(state, Epsilon)
            ) is not None:
                if eps_state not in visited:
                    visited.add(eps_state)
                    add_next_states(eps_state, next_states, visited)
            else:
                next_states.append(state)

        start = 0
        while start < len(self.text):
            current = []
            add_next_states(self.dfa.start_state, current, set())
            char_index = start - 1
            for i in range(start, len(self.text)):
                char_index += 1
                next_states = []

                for source in current:
                    if (
                        s := self.dfa[source].match_atom(Character(self.text[i]), None)
                    ) is not None:
                        add_next_states(s, next_states, set())

                if not next_states:
                    break
                current = next_states

            if any(s.accepts for s in current):
                yield Match(start, char_index + 1, self.text[start : char_index + 1])
                start = char_index + 1
            else:
                start = start + 1

    def __repr__(self):
        return f"{self.__class__.__name__}(regex={self.regexp!r}, text={self.text!r})"


if __name__ == "__main__":
    # regex = "[^A-Za-z_](ab)?b+"
    # t = "5abb"  # nfa.draw_with_graphviz()
    # regex = r"a*b+a.a*b|d+[A-Z]?"
    # t = "aaabbaaabbbaadA"
    # matcher = Matcher(t, regex)
    #
    # print(matcher)
    # for span in matcher:
    #     print(span)
    #
    # for span in re.finditer(regex, t):
    #     print(span)

    regex = r".*"
    t = "aaabbaaabbbaadA"
    matcher = Matcher(t, regex)

    print(matcher)
    for span in matcher:
        print(span)

    for span in re.finditer(regex, t):
        print(span)
