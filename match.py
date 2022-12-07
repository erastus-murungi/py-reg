import re
from dataclasses import dataclass
from parser import Anchor

from core import State
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

    def search_from_index(self, current_state: State, current_index: int):
        if current_state is not None:
            matches = self.compiled_regex[current_state].match_atom(
                self.text, current_index, None
            )
            indices = []
            for sym, next_state in matches:
                current_state = next_state

                index = self.search_from_index(
                    next_state,
                    current_index if isinstance(sym, Anchor) else current_index + 1,
                )

                if index is not None:
                    indices.append(index)

            if indices:
                return max(indices)

            if current_state.accepts:
                return current_index if not matches else current_index + 1
        return None

    def __iter__(self):
        start_index = 0
        while start_index < len(self.text):
            end_index = self.search_from_index(
                self.compiled_regex.start_state, start_index
            )
            if end_index is not None:
                yield Match(start_index, end_index, self.text[start_index:end_index])
                # empty matches
                if end_index == start_index:
                    start_index = end_index + 1
                else:
                    start_index = end_index
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
    regex, t = ("(abc)\\1", "abcabc")

    for span in re.finditer(regex, t):
        print(span)

    matcher = RegexMatcher(regex, t)

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
