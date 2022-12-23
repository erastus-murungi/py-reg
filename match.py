import re
from dataclasses import dataclass
from operator import itemgetter
from typing import Collection, Optional

from more_itertools import first_true, flatten

from core import DFA, NFA, RegexFlag, RegexParser, State, Transition, Virtual


def all_min(items: Collection):
    minimum = min(items, key=itemgetter(0))[0]
    return minimum, list(flatten([value for key, value in items]))


def all_max(items: Collection):
    maximum = max(items, key=itemgetter(0))[0]
    return maximum, list(flatten([value for key, value in items]))


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


@dataclass(frozen=True, slots=True)
class Match:
    start: int
    end: int
    text: str
    captured_groups: list[tuple[CapturedGroup]]

    @property
    def span(self):
        return self.start, self.end

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(span={self.span}, "
            f"match={self.text[self.start:self.end]!r})"
        )

    def all_groups(self):
        return tuple(self.groups(index) for index in range(len(self.captured_groups)))

    def groups(self, index=0) -> tuple[str, ...]:
        return tuple(
            captured_group.string(self.text)
            for captured_group in self.captured_groups[index]
        )

    def group(self, index=0, which=0) -> Optional[str]:
        if index < 0 or index > len(self.captured_groups[index]):
            raise IndexError(
                f"index should be 0 <= {len(self.captured_groups[which][index]) + 1}"
            )
        if index == 0:
            return self.text[self.start : self.end]
        return self.captured_groups[which][index - 1].string(self.text)


class Regexp(DFA):
    def __init__(self, pattern: str):
        self.pattern = pattern
        self.parser = RegexParser(pattern)
        nfa = NFA()
        nfa.set_terminals(self.parser.root.fsm(nfa))
        nfa.update_symbols_and_states()
        super().__init__(nfa=nfa)
        self.minimize()

    def _match(
        self, state: State, text: str, position: int, flags: RegexFlag
    ) -> list[Transition]:
        return [
            transition
            for transition in self[state]
            if transition.match(text, position, flags) is not None
        ]

    def _try_match_from_index(
        self, text: str, state: State, index: int, captured_groups: tuple[CapturedGroup]
    ) -> Optional[tuple[int, list[tuple[CapturedGroup, ...]]]]:
        if state is not None:
            matching_indices = []

            if state in self.accept:
                matching_indices.append((index, [captured_groups]))

            transitions = self._match(state, text, index, self.parser.flags)

            for matchable, end_state in transitions:
                captured_groups_copy = tuple(
                    captured_group.copy() for captured_group in captured_groups
                )

                if matchable.opening_group():
                    captured_groups_copy[matchable.group_index].start = index

                elif matchable.closing_group():
                    captured_groups_copy[matchable.group_index].end = index

                result = self._try_match_from_index(
                    text,
                    end_state,
                    index + (not isinstance(matchable, Virtual)),
                    captured_groups_copy,
                )

                if result is not None:
                    matching_indices.append(result)

            if matching_indices:
                return (
                    all_min(matching_indices)
                    if state in self.lazy
                    else all_max(matching_indices)
                )

        return None

    def match(self, text):
        """Try to apply the pattern at the start of the string, returning
        a Match object, or None if no match was found."""
        return first_true(self.finditer(text), default=None)

    def finditer(self, text: str):
        index = 0
        while index <= len(text):
            captured_groups = tuple(
                CapturedGroup() for _ in range(self.parser.group_count)
            )
            if (
                result := self._try_match_from_index(
                    text, self.start, index, captured_groups
                )
            ) is not None:
                position, captured_groups = result
                yield Match(
                    index,
                    position,
                    text,
                    captured_groups,
                )
                index = position + 1 if position == index else position
            else:
                index = index + 1

    def findall(self, text):
        return list(self.finditer(text))

    def __repr__(self):
        return f"{self.__class__.__name__}(regex={self.pattern!r})"


if __name__ == "__main__":
    # regex, t = ("ab{0,}bc", "abbbbc")
    # regex, t = ("((a)*|b)(ab|b)", "aaab")
    # regex, t = ("(a|bcdef|g|ab|c|d|e|efg|fg)*", "abcdefg")
    regex, t = ("()ef", "def")
    print(Regexp(regex).findall(t))
    print(list(re.finditer(regex, t)))

    # for span in re.finditer(regex, t):
    #     print(span)

    # print(re.match(regex, t))
    # print(Regexp(regex).match(t))

    print([m.groups() for m in re.finditer(regex, t)])
    print(Regexp(regex).match(t).all_groups())

    # for span in matcher:
    #     print(span.groups())

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
