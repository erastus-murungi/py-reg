import re
from dataclasses import dataclass

from pyreg import compile_regex


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
        self.compiled_regex = compile_regex(regexp)
        self.compiled_regex.draw_with_graphviz()

    def search_from(self, start_index) -> tuple[Match, int]:
        current_state = self.compiled_regex.start_state
        accepting_indices: list[int] = []

        for current_index in range(start_index, len(self.text)):
            if (
                next_state := self.compiled_regex[current_state].match_atom(
                    self.text, current_index, None
                )
            ) is None:
                break

            current_state = next_state
            if current_state.accepts:
                accepting_indices.append(current_index)

        if accepting_indices:
            index = accepting_indices[-1]
            match = Match(start_index, index + 1, self.text[start_index : index + 1])
            start_index = index + 1
        else:
            start_index = start_index + 1
            match = None
        return match, start_index

    def __iter__(self):
        start_index = 0
        while start_index < len(self.text):
            match, start_index = self.search_from(start_index)
            if match:
                yield match

    def __repr__(self):
        return f"{self.__class__.__name__}(regex={self.regexp!r}, text={self.text!r})"


if __name__ == "__main__":
    # regex = c
    # t = "5abb"  # nfa.draw_with_graphviz()
    regex = r"^a*b+a.a*b|d+[A-Z]?(CD)+"
    t = "aaabbaaabbbaadACDC"
    matcher = Matcher(t, regex)

    print(matcher)
    for span in matcher:
        print(span)

    for span in re.finditer(regex, t):
        print(span)

    # regex = r".*"
    # t = "aaabbaaabbbaadA"
    # matcher = Matcher(t, regex)
    #
    # print(matcher)
    # for span in matcher:
    #     print(span)
    #
    # for span in re.finditer(regex, t):
    #     print(span)
