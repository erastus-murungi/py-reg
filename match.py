import re
from dataclasses import dataclass
from parser import Anchor
from typing import Optional

from core import RegexFlag, State
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
        self.flags = RegexFlag.NOFLAG

    def _try_match_from_index(self, state: State, position, flags) -> Optional[int]:
        if state is not None:
            matching_indices = []

            if state.accepts:
                matching_indices.append(position)

            matches = self.compiled_regex[state].match(self.text, position, flags)

            for symbol, next_state in matches:
                index = self._try_match_from_index(
                    next_state, position + (not isinstance(symbol, Anchor)), self.flags
                )
                if index is not None:
                    matching_indices.append(index)

            if matching_indices:
                return min(matching_indices) if state.lazy else max(matching_indices)

        return None

    def __iter__(self):
        index = 0
        while index <= len(self.text):
            position = self._try_match_from_index(
                self.compiled_regex.start_state, index, self.flags
            )
            if position is not None:
                yield Match(index, position, self.text[index:position])
                index = position + 1 if position == index else position
            else:
                index = index + 1

    def __repr__(self):
        return f"{self.__class__.__name__}(regex={self.regexp!r}, text={self.text!r})"


if __name__ == "__main__":
    regex, t = ("a{2,3}?", "aaaaa")
    matcher = RegexMatcher(regex, t)

    for span in re.finditer(regex, t):
        print(span)

    print(matcher)
    for span in matcher:
        print(span)
