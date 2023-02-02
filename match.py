import re
from dataclasses import dataclass
from parser import RegexpParser
from pprint import pprint
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


CapturedGroups = list[CapturedGroup]
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
            raise IndexError(f"index should be 0 <= {len(self.captured_groups)}")
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

    @staticmethod
    def _update_captured_groups(
        index: int, matchable: Tag, captured_groups: CapturedGroups
    ) -> CapturedGroups:
        if matchable.is_opening_group() or matchable.is_closing_group():
            group_index = matchable.group_index

            # must create copy of the list
            groups_copy = captured_groups[:]

            # copy actual group object
            captured_group_copy = captured_groups[group_index].copy()
            if matchable.is_opening_group():
                captured_group_copy.start = index
            else:
                captured_group_copy.end = index

            groups_copy[group_index] = captured_group_copy
            return groups_copy
        return captured_groups

    def _match_at_index_with_groups(
        self,
        text: str,
        index: int,
    ) -> Optional[MatchResult]:

        captured_groups = [CapturedGroup() for _ in range(self.parser.group_count)]

        # we only need to keep track of 3 state variables
        work_list = [(self.start, index, captured_groups, ())]

        while work_list:
            current_state, index, captured_groups, path = work_list.pop()

            if current_state in self.accept:
                return index, captured_groups

            for matchable, end_state in reversed(self.step(current_state, text, index)):
                if (current_state, end_state, index) in path:
                    continue

                path_copy = path + ((current_state, end_state, index),)
                updated_captured_groups = self._update_captured_groups(
                    index, matchable, captured_groups
                )
                work_list.append(
                    (
                        end_state,
                        matchable.increment(index),
                        updated_captured_groups,
                        path_copy,
                    )
                )

        return None

    def _match_at_index_no_groups(
        self,
        text: str,
        index: int,
    ) -> Optional[int]:
        # we only need to keep track of 2 state variables
        work_list = [(self.start, index, ())]

        while work_list:
            current_state, index, path = work_list.pop()

            if current_state in self.accept:
                return index

            work_list.extend(
                (
                    end_state,
                    matchable.increment(index),
                    path + ((current_state, end_state, index),),
                )
                for matchable, end_state in reversed(
                    self.step(current_state, text, index)
                )
                if (current_state, end_state, index) not in path
            )

        return None

    def _match_at_index(self, text: str, index: int) -> Optional[MatchResult]:
        if self.parser.group_count > 0:
            return self._match_at_index_with_groups(text, index)
        else:
            if (position := self._match_at_index_no_groups(text, index)) is not None:
                return position, []
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
        return [m.group(0) for m in self.finditer(text)]

    def __repr__(self):
        return f"{self.__class__.__name__}(regex={self.pattern!r})"


if __name__ == "__main__":
    regex, t = "[-a-zA-Z0-9@:%._\\+~#=]", "foo"

    print(list(re.finditer(regex, t)))
    print([m.groups() for m in re.finditer(regex, t)])

    pattern = Regexp(regex)
    # pattern.graph()
    # DFA(pattern).graph()
    pprint(list(pattern.finditer(t)))

    print([m.groups() for m in Regexp(regex).finditer(t)])
