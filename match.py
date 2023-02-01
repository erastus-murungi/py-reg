import re
from dataclasses import dataclass
from parser import RegexpParser, Virtual
from pprint import pprint
from time import monotonic
from typing import Optional

from more_itertools import first_true

from core import NFA, State, Tag, Transition


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


CapturedGroups = tuple[CapturedGroup, ...]
MatchResult = tuple[int, CapturedGroups]


@dataclass(frozen=True, slots=True)
class Match:
    start: int
    end: int
    text: str
    captured_groups: CapturedGroups

    @property
    def span(self):
        return self.start, self.end

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(span={self.span}, "
            f"match={self.text[self.start:self.end]!r})"
        )

    def groups(self) -> tuple[str, ...]:
        return tuple(
            captured_group.string(self.text) for captured_group in self.captured_groups
        )

    def group(self, index=0) -> Optional[str]:
        if index < 0 or index > len(self.captured_groups):
            raise IndexError(
                f"index should be 0 <= {len(self.captured_groups[index]) + 1}"
            )
        if index == 0:
            return self.text[self.start : self.end]
        return self.captured_groups[index - 1].string(self.text)


class Regexp(NFA):
    def __init__(self, regexp: str):
        super().__init__()
        self.pattern = regexp
        self.parser = RegexpParser(regexp)
        self.set_terminals(self.parser.root.accept(self))
        self.update_symbols_and_states()

    def matches(self, state, text, index):
        for transition in self[state]:
            if (
                transition.matchable == Tag.epsilon() and transition.end in self.accept
            ) or (
                transition.matchable != Tag.epsilon()
                and transition.match(text, index, self.parser.flags) is not None
            ):
                yield transition

    def _compute_step(
        self, state: State, text: str, index: int
    ) -> tuple[list[State], list[Transition]]:
        explored = set()
        stack = [(False, state)]
        closure = []
        transitions = []

        while stack:
            completed, state = stack.pop()
            if completed:
                closure.append(state)
                # once we are done with this state
                transitions.extend(self.matches(state, text, index))

            if state in explored:
                continue

            explored.add(state)

            stack.append((True, state))
            # explore the states in the order which they are in
            stack.extend(
                (False, nxt)
                for nxt in self.transition(state, Tag.epsilon(), True)[::-1]
            )
        return closure, transitions

    def step(self, state: State, text: str, index: int) -> list[Transition]:
        _, transitions = self._compute_step(state, text, index)
        return transitions

    def _match_at_index_with_groups(
        self,
        text: str,
        index: int,
    ) -> Optional[MatchResult]:

        captured_groups = [CapturedGroup() for _ in range(self.parser.group_count)]

        # we only need to keep track of 3 state variables
        work_list = [(self.start, index, captured_groups)]

        while work_list:
            current_state, index, captured_groups = work_list.pop()

            if current_state in self.accept:
                return index, captured_groups

            for matchable, end_state in reversed(self.step(current_state, text, index)):
                # only create a copy of captured groups when a modification is made
                if matchable.is_opening_group() or matchable.is_closing_group():
                    group_index = matchable.group_index
                    captured_group_copy = captured_groups[group_index].copy()
                    if matchable.is_opening_group():
                        captured_group_copy.start = index
                    else:
                        captured_group_copy.end = index
                    # must create a copy of the list
                    captured_groups = captured_groups[:]
                    captured_groups[group_index] = captured_group_copy

                work_list.append(
                    (
                        end_state,
                        index + (not isinstance(matchable, Virtual)),
                        captured_groups,
                    )
                )

        return None

    def _match_at_index_no_groups(
        self,
        text: str,
        index: int,
    ) -> Optional[int]:
        # we only need to keep track of 2 state variables
        work_list = [(self.start, index)]

        while work_list:
            current_state, index = work_list.pop()

            if current_state in self.accept:
                return index

            work_list.extend(
                (end_state, index + (not isinstance(matchable, Virtual)))
                for matchable, end_state in reversed(
                    self.step(current_state, text, index)
                )
            )

        return None

    def _match_at_index(self, text: str, index: int) -> Optional[MatchResult]:
        if self.parser.group_count > 0:
            return self._match_at_index_with_groups(text, index)
        else:
            if (position := self._match_at_index_no_groups(text, index)) is not None:
                return position, ()
            return None

    def match(self, text: str):
        """Try to apply the pattern at the start of the string, returning
        a Match object, or None if no match was found."""
        return first_true(self.finditer(text), default=None)

    def finditer(self, text: str):
        index = 0
        while index <= len(text):
            if (result := self._match_at_index(text, index)) is not None:
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
    # regex, t = ("(0(_?0)*|[1-9](_?[0-9])*)", "17429")
    # regex, t = (r"(a?)((ab)?)", "ab")
    # regex, t = r"^ab|(abab)$", "abbabab"
    # regex, t = "a.+?c", "abcabc"
    # regex, t = "a?", "a"
    # regex, t = r"([0a-z][a-z0-9]*,)+", r"a5,b7,c9,"
    regex, t = "a*(^a)", "aa"
    # regex, t = "(?:ab)+", "ababa"
    # regex, t = "(a*)*", "-",
    # regex, t = ("([^.]*)\\.([^:]*):[T ]+(.*)", "track1.title:TBlah blah blah")

    # regex, t = "(?i)(a+|b){0,1}?", "AB"
    pattern = Regexp(regex)
    pattern.graph()
    # DFA(pattern).graph()
    start = monotonic()
    pprint(pattern.findall(t))
    print(f"findall took {monotonic() - start} seconds ...")
    pprint(list(re.finditer(regex, t)))

    # for span in re.finditer(regex, t):
    #     print(span)

    # print(re.match(regex, t))
    # print(Regexp(regex).match(t))

    print([m.groups() for m in re.finditer(regex, t)])
    print(pattern.match(t).group())

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
