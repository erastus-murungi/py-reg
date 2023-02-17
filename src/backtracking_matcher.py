from typing import Optional

from src.core import CapturedGroup, CapturedGroups, MatchResult, RegexPattern
from src.fsm import DFA, NFA, State, Transition
from src.parser import EPSILON, Anchor, AnchorType, Matcher, RegexFlag, RegexParser


class RegexNFA(NFA, RegexPattern):
    def __init__(self, pattern: str, flags: RegexFlag = RegexFlag.NOFLAG):
        super().__init__()
        self._parser = RegexParser(pattern, flags)
        self.set_terminals(self._parser.root.accept(self))
        self.update_symbols_and_states()

    def recover(self) -> str:
        return self._parser.root.string()

    def _match_suffix_dfa(self, state: State, text: str, index: int) -> Optional[int]:
        """
        This a fast matcher when you don't have groups or greedy quantifiers
        """
        assert self._parser.group_count == 0

        if state is not None:
            matching_indices = []

            if state in self.accepting_states:
                matching_indices.append(index)

            transitions = [
                transition
                for transition in self[state]
                if transition.matcher(text, index, self._parser.flags)
            ]

            for matchable, end_state in transitions:
                result = self._match_suffix_dfa(
                    end_state,
                    text,
                    matchable.increment(index),
                )

                if result is not None:
                    matching_indices.append(result)

            if matching_indices:
                return max(matching_indices)

        return None

    def step(self, start_state: State, text: str, index: int) -> list[Transition]:
        """
        Performs a depth first search to collect valid transitions the transitions reachable through epsilon transitions
        """
        explored: set[State] = set()
        stack: list[tuple[bool, State]] = [(False, start_state)]
        transitions: list[Transition] = []

        while stack:
            completed, state = stack.pop()
            if completed:
                # we can easily compute the close by append state to a closure
                # collection i.e `closure.append(state)`
                # once we are done with this state
                for transition in self[state]:
                    # augment to match epsilon transitions which lead to accepting states
                    # we could rewrite things by passing more context into the match method so that
                    # this becomes just a one-liner: transition.matcher(context)
                    if (
                        transition.matcher is EPSILON
                        and transition.end in self.accepting_states
                    ) or (transition.matcher(text, index, self._parser.flags)):
                        transitions.append(transition)

            if state in explored:
                continue

            explored.add(state)

            stack.append((True, state))
            # explore the states in the order which they are in
            stack.extend(
                (False, nxt) for nxt in self.transition(state, EPSILON, True)[::-1]
            )
        return transitions

    @staticmethod
    def _update_captured_groups(
        index: int, matchable: Matcher, captured_groups: CapturedGroups
    ) -> CapturedGroups:
        if isinstance(matchable, Anchor) and matchable.anchor_type in (
            AnchorType.GroupEntry,
            AnchorType.GroupExit,
        ):
            group_index = matchable.group_index
            # must create copy of the list
            groups_copy = captured_groups[:]
            # copy actual group object
            captured_group_copy = captured_groups[group_index].copy()
            if matchable.anchor_type == AnchorType.GroupEntry:
                captured_group_copy.start = index
            else:
                captured_group_copy.end = index

            groups_copy[group_index] = captured_group_copy
            return groups_copy
        return captured_groups

    def _match_suffix_with_groups(
        self,
        text: str,
        index: int,
    ) -> Optional[MatchResult]:

        captured_groups = [CapturedGroup() for _ in range(self._parser.group_count)]

        # we only need to keep track of 3 state variables
        work_list = [(self.start_state, index, captured_groups, ())]

        while work_list:
            current_state, index, captured_groups, path = work_list.pop()

            if current_state in self.accepting_states:
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

    def _match_suffix_no_groups(
        self,
        text: str,
        index: int,
    ) -> Optional[int]:
        # we only need to keep track of 2 state variables
        work_list = [(self.start_state, index, ())]

        while work_list:
            current_state, index, path = work_list.pop()

            if current_state in self.accepting_states:
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

    def match_suffix(self, text: str, index: int) -> Optional[MatchResult]:
        if isinstance(super(), DFA):
            if (
                position := self._match_suffix_dfa(self.start_state, text, index)
            ) is not None:
                return position, []
            return None
        if self._parser.group_count > 0:
            return self._match_suffix_with_groups(text, index)
        else:
            if (position := self._match_suffix_no_groups(text, index)) is not None:
                return position, []
            return None

    def __repr__(self):
        return super().__repr__()


if __name__ == "__main__":
    regex, t = "(a*)*", "-"

    p = RegexNFA(regex)
    import re

    print(list(re.finditer(regex, t)))
    print([m.groups() for m in re.finditer(regex, t)])
    # pattern.graph()
    print(list(p.finditer(t)))
    print([m.groups() for m in p.finditer(t)])
