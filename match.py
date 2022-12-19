import re
from dataclasses import dataclass
from operator import itemgetter
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


@dataclass(slots=True)
class CapturedGroup:
    start: Optional[int] = None
    end: Optional[int] = None

    def copy(self):
        return CapturedGroup(self.start, self.end)

    def string(self, text: str):
        if self.start is not None and self.end is not None:
            return text[self.start : self.end]
        return None


class RegexMatcher:
    def __init__(self, regexp: str, text: str):
        self.text = text
        self.regexp = regexp
        self.compiled_regex = CompiledRegex(regexp)
        # self.compiled_regex.graph()

    def gen_groups(
        self,
        start_index: int,
        end_index: Optional[int],
        captured_groups: list[CapturedGroup],
    ) -> list[Optional[str]]:
        groups = [self.text[start_index:end_index] if end_index is not None else None]
        for captured_group in captured_groups:
            groups.append(captured_group.string(self.text))
        return groups

    def _try_match_from_index(
        self, state: State, index: int, captured_groups: list[CapturedGroup]
    ) -> Optional[int]:
        if state is not None:
            matching_indices = []

            if state in self.compiled_regex.accept:
                matching_indices.append((index, captured_groups))

            transitions = self.compiled_regex.match(
                state, self.text, index, self.compiled_regex.flags
            )

            for matchable, end_state in transitions:
                groups_copy = [value.copy() for value in captured_groups]
                if matchable.opening_group():
                    groups_copy[matchable.group_index].start = index
                if matchable.closing_group():
                    groups_copy[matchable.group_index].end = index
                next_index = self._try_match_from_index(
                    end_state, index + (not isinstance(matchable, Virtual)), groups_copy
                )
                if next_index is not None:
                    matching_indices.append(next_index)

            if matching_indices:
                return (
                    min(matching_indices, key=itemgetter(0))
                    if state in self.compiled_regex.lazy
                    else max(matching_indices, key=itemgetter(0))
                )

        return None

    def __iter__(self):
        index = 0
        while index <= len(self.text):
            captured_groups = [
                CapturedGroup() for _ in range(self.compiled_regex.group_count)
            ]
            position = self._try_match_from_index(
                self.compiled_regex.start, index, captured_groups
            )
            if position is not None:
                position, path = position
                yield Match(
                    index,
                    position,
                    self.text[index:position],
                    self.gen_groups(index, position, path),
                )
                index = position + 1 if position == index else position
            else:
                index = index + 1

    def __repr__(self):
        return f"{self.__class__.__name__}(regex={self.regexp!r}, text={self.text!r})"


if __name__ == "__main__":
    # regex, t = ("ab{0,}bc", "abbbbc")
    regex, t = ("((a)*|b)(ab)", "aaab")
    matcher = RegexMatcher(regex, t)
    print(matcher)
    # matcher.compiled_regex.graph()

    # for span in re.finditer(regex, t):
    #     print(span)

    print(re.match(regex, t).groups())
    # print(re.match(regex, t).group(0))

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
