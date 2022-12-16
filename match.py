import re
from dataclasses import dataclass
from typing import Optional

from core import State, Virtual
from pyreg import CompiledRegex


@dataclass(frozen=True, slots=True)
class Match:
    start: int
    end: int
    substr: str
    _groups: []

    @property
    def span(self):
        return self.start, self.end

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(span={self.span}, "
            f"match={self.substr!r})"
        )

    def groups(self):
        return self._groups[:]

    def group(self, index):
        return self._groups[index]


class RegexMatcher:
    def __init__(self, regexp: str, text: str):
        self.text = text
        self.regexp = regexp
        self.compiled_regex = CompiledRegex(regexp)
        # self.compiled_regex.graph()

    def _try_match_from_index(self, state: State, index: int) -> Optional[int]:
        if state is not None:
            matching_indices = []

            if state in self.compiled_regex.accept:
                matching_indices.append(index)

            transitions = self.compiled_regex.match(
                state, self.text, index, self.compiled_regex.flags
            )

            for matchable, end_state in transitions:
                next_index = self._try_match_from_index(
                    end_state, index + (not isinstance(matchable, Virtual))
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

    def gen_groups(
        self,
        start_index: int,
        end_index: Optional[int],
        captured_groups,
        start2index: dict[State, int],
    ) -> list[Optional[str]]:
        groups = []
        if end_index is None:
            groups.append(None)
        else:
            groups.append(self.text[start_index:end_index])

        for group_index, group_entry in start2index.items():
            group_exit = captured_groups[group_index]
            if group_exit is None:
                groups.append(None)
            else:
                groups.append(self.text[group_entry:group_exit])

        return groups

    def __iter__(self):
        index = 0
        while index <= len(self.text):
            position = self._try_match_from_index(self.compiled_regex.start, index)
            if position is not None:
                yield Match(
                    index,
                    position,
                    self.text[index:position],
                    self.gen_groups(index, position, {}, {}),
                )
                index = position + 1 if position == index else position
            else:
                index = index + 1

    def __repr__(self):
        return f"{self.__class__.__name__}(regex={self.regexp!r}, text={self.text!r})"


if __name__ == "__main__":
    regex, t = ("((a)*|b)(ab|b)", "aaab")
    # regex, t = ('((a)*|b)', 'aaa')
    matcher = RegexMatcher(regex, t)
    print(matcher)
    matcher.compiled_regex.graph()

    # for span in re.finditer(regex, t):
    #     print(span)

    print(re.match(regex, t).groups())
    print(re.match(regex, t).group(0))

    for span in matcher:
        print(span.groups())

    # for span in RegexMatcher(regex, t):
    #     print(span)

    # # regex, t = ("a*?", "aaab")
    #
    # regex, t = (' (.*)a|(.*)b', 'aaaaab')
    # matcher = RegexMatcher(regex, t)
    # matcher.compiled_regex.graph()
    #
    # for span in re.finditer(regex, t):
    #     print(span)
    #
    # print(re.match(regex, t).groups())
    #
    # for span in matcher:
    #     print(span)
