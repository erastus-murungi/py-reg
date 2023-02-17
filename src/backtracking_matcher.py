from operator import attrgetter
from typing import Optional

from src.fsm import DFA, NFA, State, Transition
from src.matching import Cursor, RegexPattern
from src.parser import EPSILON, RegexFlag, RegexParser


class RegexNFA(NFA, RegexPattern):
    def __init__(self, pattern: str, flags: RegexFlag = RegexFlag.NOFLAG):
        NFA.__init__(self)
        RegexPattern.__init__(self, RegexParser(pattern, flags))
        self.set_terminals(self.parser.root.accept(self))
        self.update_symbols_and_states()

    def recover(self) -> str:
        return self.parser.root.string()

    def _match_suffix_dfa(self, state: State, cursor: Cursor) -> Optional[int]:
        """
        This a fast matcher when you don't have groups or greedy quantifiers
        """
        assert self.parser.group_count == 0

        if state is not None:
            matching_cursors = []

            if state in self.accepting_states:
                matching_cursors.append(cursor)

            transitions = [
                transition for transition in self[state] if transition.matcher(cursor)
            ]

            for matcher, end_state in transitions:
                result = self._match_suffix_dfa(
                    end_state,
                    matcher.update_index(cursor),
                )

                if result is not None:
                    matching_cursors.append(result)

            if matching_cursors:
                return max(matching_cursors, key=attrgetter("position"))

        return None

    def step(self, start_state: State, cursor: Cursor) -> list[Transition]:
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
                    if transition.matcher(cursor) or (
                        transition.matcher is EPSILON
                        and transition.end in self.accepting_states
                    ):
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

    def _match_suffix_with_groups(
        self,
        cursor: Cursor,
    ) -> Optional[Cursor]:
        # we only need to keep track of 3 state variables
        stack = [(self.start_state, cursor, ())]

        while stack:
            state, cursor, path = stack.pop()

            if state in self.accepting_states:
                return cursor

            for matcher, end_state in reversed(self.step(state, cursor)):
                if (state, end_state, cursor.position) in path:
                    continue

                path_copy = path + ((state, end_state, cursor.position),)
                stack.append(
                    (
                        end_state,
                        matcher.update(cursor),
                        path_copy,
                    )
                )

        return None

    def _match_suffix_no_groups(
        self,
        cursor: Cursor,
    ) -> Optional[int]:
        # we only need to keep track of 2 state variables
        stack = [(self.start_state, cursor, ())]

        while stack:
            state, cursor, path = stack.pop()

            if state in self.accepting_states:
                return cursor

            stack.extend(
                (
                    end_state,
                    matcher.update(cursor),
                    path + ((state, end_state, cursor.position),),
                )
                for matcher, end_state in reversed(self.step(state, cursor))
                if (state, end_state, cursor.position) not in path
            )

        return None

    def match_suffix(self, cursor: Cursor) -> Optional[Cursor]:
        if isinstance(super(), DFA):
            return self._match_suffix_dfa(self.start_state, cursor)
        elif self.parser.group_count > 0:
            return self._match_suffix_with_groups(cursor)
        else:
            return self._match_suffix_no_groups(cursor)

    def __repr__(self):
        return super().__repr__()


if __name__ == "__main__":
    regex, t = "ab", "abab"

    p = RegexNFA(regex)
    import re

    print(list(re.finditer(regex, t)))
    print([m.groups() for m in re.finditer(regex, t)])
    # p.graph()
    print(list(p.finditer(t)))
    print([m.groups() for m in p.finditer(t)])
