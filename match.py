import re
from dataclasses import dataclass
from typing import Optional

from core import Anchor, RegexFlag, State
from pyreg import CompiledRegex


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
        self.compiled_regex = CompiledRegex(regexp)
        self.flags = RegexFlag.NOFLAG

    def _try_match_from_index(self, state: State, index: int) -> Optional[int]:
        if state is not None:
            matching_indices = []

            if state in self.compiled_regex.accept:
                matching_indices.append(index)

            transitions = self.compiled_regex.match(state, self.text, index, self.flags)

            for matchable, end_state in transitions:
                next_index = self._try_match_from_index(
                    end_state, index + (not isinstance(matchable, Anchor))
                )
                if next_index is not None:
                    matching_indices.append(next_index)

            if matching_indices:
                return (
                    min(matching_indices)
                    if state in self.compiled_regex.lazy
                    else max(matching_indices)
                )

        return None

    def __iter__(self):
        index = 0
        while index <= len(self.text):
            position = self._try_match_from_index(self.compiled_regex.start, index)
            if position is not None:
                yield Match(index, position, self.text[index:position])
                index = position + 1 if position == index else position
            else:
                index = index + 1

    def __repr__(self):
        return f"{self.__class__.__name__}(regex={self.regexp!r}, text={self.text!r})"


if __name__ == "__main__":
    regex, t = (
        "([0-9](_?[0-9])*\\.([0-9](_?[0-9])*)?|\\.[0-9](_?[0-9])*)([eE][-+]?[0-9](_?[0-9])*)?",
        "0.1",
    )
    matcher = RegexMatcher(regex, t)

    for span in re.finditer(regex, t):
        print(span)

    for span in matcher:
        print(span)

    print(matcher)
