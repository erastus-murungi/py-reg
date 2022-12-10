import re
from dataclasses import dataclass
from parser import Anchor
from typing import Optional

from core import RegexContext, State
from pyreg import compile_regex


@dataclass(frozen=True, slots=True)
class Match:
    start: int
    end: int
    substr: str

    @property
    def span(self):
        return self.start, self.end

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(span={self.span}, "
            f"match={self.substr!r})"
        )


class RegexMatcher:
    def __init__(self, regexp: str, text: str):
        self.text = text
        self.regexp = regexp
        self.compiled_regex = compile_regex(regexp)

    def search_from_index(
        self, current_state: State, context: RegexContext
    ) -> Optional[int]:
        if current_state is not None:
            matches = self.compiled_regex[current_state].match_atom(context)
            indices = []
            for sym, next_state in matches:
                current_state = next_state

                index = self.search_from_index(
                    next_state,
                    context.copy() if isinstance(sym, Anchor) else context.increment(),
                )

                if index is not None:
                    indices.append(index)

            if indices:
                return max(indices)

            if current_state.accepts:
                return context.position if not matches else context.position + 1
        return None

    def __iter__(self):
        start_index = 0
        while start_index <= len(self.text):
            position = self.search_from_index(
                self.compiled_regex.start_state, RegexContext(self.text, start_index)
            )
            if position is not None:
                yield Match(start_index, position, self.text[start_index:position])
                if position == start_index:
                    start_index = position + 1
                else:
                    start_index = position
            else:
                start_index = start_index + 1

    def __repr__(self):
        return f"{self.__class__.__name__}(regex={self.regexp!r}, text={self.text!r})"


if __name__ == "__main__":
    # regex = c
    # t = "5abb"  # nfa.draw_with_graphviz()
    # regex = r"^a*b+a.a*b|d+[A-Z]?(CD)+\w\d+r$"
    # t = "aaabbaaabbbaadACDC075854r"
    # regex = r"(ab){3,8}"
    # t = "abababababab"
    # regex, t = (r"h.*od?", "hello\ngoodbye\n")
    # regex, t = r'a', 'a'
    # regex, t = (r"[abcd]+", "xxxabcdxxx")

    regex, t = (".*Python", "Python")
    matcher = RegexMatcher(regex, t)

    for span in re.finditer(regex, t):
        print(span)

    print(matcher)
    for span in matcher:
        print(span)

    # regex = r".*"
    # t = "aaabbaaabbbaadA"
    # matcher = Matcher(t, regex)x
    #
    # print(matcher)
    # for span in matcher:
    #     print(span)
    #
    # for span in re.finditer(regex, t):
    #     print(span)
