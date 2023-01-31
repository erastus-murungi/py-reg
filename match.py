import re
from dataclasses import dataclass
from itertools import chain
from pprint import pprint
from typing import Collection, Optional
from time import monotonic

from more_itertools import first_true

from core import DFA, NFA, RegexFlag, RegexpParser, State, Tag, Transition, Virtual


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
MatchResult = tuple[int, list[CapturedGroups]]


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

    def all_groups(self):
        return tuple(self.groups(index) for index in range(len(self.captured_groups)))

    def groups(self, index: int = 0) -> tuple[str, ...]:
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


class Regexp(NFA):
    def __init__(self, regexp: str):
        super().__init__()
        self.pattern = regexp
        self.parser = RegexpParser(regexp)
        self.set_terminals(self.parser.root.fsm(self))
        self.update_symbols_and_states()

    def _match(
        self, state: State, text: str, position: int, flags: RegexFlag
    ) -> list[Transition]:
        return [
            transition
            for transition in self[state]
            if transition.match(text, position, flags) is not None
        ]

    def step(
        self, states: Collection[State], text, index
    ) -> tuple[list[State], list[Transition]]:

        explored = set()
        stack = [(False, state) for state in states]
        closure = []
        transitions = []

        while stack:
            completed, state = stack.pop()
            if completed:
                closure.append(state)
                # once we are done with this state
                for transition in self[state]:
                    if (
                        transition.matchable == Tag.epsilon()
                        and transition.end in self.accept
                    ) or (
                        transition.matchable != Tag.epsilon()
                        and transition.match(text, index, self.parser.flags)
                    ):
                        transitions.append(transition)

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

    def _match_at_index(
        self, text: str, state: State, index: int, captured_groups: CapturedGroups
    ) -> Optional[MatchResult]:

        if state in self.accept:
            return index, [captured_groups]

        states, transitions = self.step([state], text, index)
        # print(states, transitions)

        for matchable, end_state in transitions:
            captured_groups_copy = tuple(
                captured_group.copy() for captured_group in captured_groups
            )

            if matchable.is_opening_group():
                captured_groups_copy[matchable.group_index].start = index

            elif matchable.is_closing_group():
                captured_groups_copy[matchable.group_index].end = index

            result = self._match_at_index(
                text,
                end_state,
                index + (not isinstance(matchable, Virtual)),
                captured_groups_copy,
            )

            if result is not None:
                return result

        return None

    def match(self, text: str):
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
                result := self._match_at_index(text, self.start, index, captured_groups)
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
    # regex, t = ("(0(_?0)*|[1-9](_?[0-9])*)", "17429")
    # regex, t = (r"(a?)((ab)?)", "ab")
    # regex, t = r"^ab|(abab)$", "abbabab"
    # regex, t = "a.+?c", "abcabc"
    # regex, t = "a?", "a"
    # regex, t = r"([0a-z][a-z0-9]*,)+", r"a5,b7,c9,"
    # regex, t = "(?:ab)+", "ababa"
    # regex, t = "(a*)*", "-",
    regex, t = ("([^.]*)\\.([^:]*):[T ]+(.*)", "track1.title:TBlah blah blah")

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
    print(pattern.match(t).all_groups())

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
