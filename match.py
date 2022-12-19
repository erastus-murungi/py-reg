import re
from dataclasses import dataclass
from operator import itemgetter
from typing import Optional

from more_itertools import first_true

from core import DFA, NFA, RegexFlag, RegexParser, State, Transition, Virtual
from simplify import simplify


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
        return self._groups[1:]

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


class Regexp(DFA):
    def __init__(self, pattern: str):
        self.pattern = pattern
        self.parser = RegexParser(simplify(pattern))
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

    @staticmethod
    def gen_groups(
        text,
        start_index: int,
        end_index: Optional[int],
        captured_groups: list[CapturedGroup],
    ) -> list[Optional[str]]:
        groups = [text[start_index:end_index] if end_index is not None else None]
        for captured_group in captured_groups:
            groups.append(captured_group.string(text))
        return groups

    def _try_match_from_index(
        self, text: str, state: State, index: int, captured_groups: list[CapturedGroup]
    ) -> Optional[tuple[int, list[CapturedGroup]]]:
        if state is not None:
            matching_indices = []

            if state in self.accept:
                matching_indices.append((index, captured_groups))

            transitions = self._match(state, text, index, self.parser.flags)

            for matchable, end_state in transitions:
                captured_groups_copy = [
                    captured_group.copy() for captured_group in captured_groups
                ]

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
                    min(matching_indices, key=itemgetter(0))
                    if state in self.lazy
                    else max(matching_indices, key=itemgetter(0))
                )

        return None

    def match(self, text):
        """Try to apply the pattern at the start of the string, returning
        a Match object, or None if no match was found."""
        return first_true(self.finditer(text), default=None)

    def finditer(self, text: str):
        index = 0
        while index <= len(text):
            captured_groups = [CapturedGroup() for _ in range(self.parser.group_count)]
            if (
                result := self._try_match_from_index(
                    text, self.start, index, captured_groups
                )
            ) is not None:
                position, captured_groups = result
                yield Match(
                    index,
                    position,
                    text[index:position],
                    self.gen_groups(text, index, position, captured_groups),
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
    # regex, t = ("((a)*|b)(ab)", "aaab")
    regex, t = ("(a|bcdef|g|ab|c|d|e|efg|fg)*", "abcdefg")
    print(Regexp(regex).findall(t))
    print(list(re.finditer(regex, t)))

    # for span in re.finditer(regex, t):
    #     print(span)

    print(re.match(regex, t))
    print(Regexp(regex).match(t))
    # print(re.match(regex, t).group(0))

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
