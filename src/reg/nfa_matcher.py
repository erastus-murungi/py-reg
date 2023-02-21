from collections import deque
from operator import itemgetter
from typing import Optional

from reg.fsm import DFA, NFA, State, Transition
from reg.matcher import Context, Cursor, RegexPattern
from reg.optimizer import Optimizer
from reg.parser import EPSILON, Anchor, RegexFlag, RegexParser


class RegexNFA(NFA, RegexPattern):
    """
    A backtracking NFA based regex pattern matcher

    Examples
    --------
    >>> pattern, text = '(ab)+', 'abab'
    >>> compiled_regex = RegexNFA(pattern)
    >>> print(list(compiled_regex.finditer(text)))
    [RegexMatch(span=(0, 4), match='abab')]
    >>> print([m.groups() for m in compiled_regex.finditer(text)])
    [('ab',)]
    """

    def __init__(self, pattern: str, flags: RegexFlag = RegexFlag.OPTIMIZE):
        NFA.__init__(self)
        RegexPattern.__init__(self, RegexParser(pattern, flags))
        if RegexFlag.OPTIMIZE & self.parser.flags:
            Optimizer.run(self.parser.root)
        self.set_terminals(self.parser.root.accept(self))
        self.update_symbols_and_states()
        self.reduce_epsilons()

    def recover(self) -> str:
        return self.parser.root.to_string()

    def _match_suffix_dfa(
        self, state: State, cursor: Cursor, context: Context
    ) -> Optional[int]:
        """
        This a fast matcher when you don't have groups or greedy quantifiers
        """
        assert self.parser.group_count == 0

        if state is not None:
            matching_cursors = []

            if state in self.accepting_states:
                matching_cursors.append(cursor)

            transitions = [
                transition
                for transition in self[state]
                if transition.matcher(cursor, context)
            ]

            for matcher, end_state in transitions:
                result = self._match_suffix_dfa(
                    end_state, matcher.update_index(cursor), context
                )

                if result is not None:
                    matching_cursors.append(result)

            if matching_cursors:
                return max(matching_cursors, key=itemgetter(0))

        return None

    def step(
        self, start_state: State, cursor: Cursor, context: Context
    ) -> list[Transition]:
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
                    if transition.matcher(cursor, context) or (
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

    def queue_transition(
        self,
        start: Transition,
        cursor: Cursor,
        context: Context,
        explored: set[tuple[int, Transition]],
    ) -> list[tuple[Transition, Cursor]]:
        """
        Performs a depth first search to collect valid transitions the transitions reachable through epsilon transitions
        """
        stack = [(nxt, start.matcher.update(cursor)) for nxt in self[start.end][::-1]]
        transitions: list[tuple[Transition, Cursor]] = []

        while stack:
            transition, cursor = stack.pop()

            if (cursor.position, transition) in explored:
                continue

            explored.add((cursor.position, transition))

            if isinstance(transition.matcher, Anchor):
                if transition.matcher is EPSILON or transition.matcher(cursor, context):
                    if transition.end in self.accepting_states:
                        transitions.append((transition, cursor))
                    else:
                        stack.extend(
                            (nxt, transition.matcher.update(cursor))
                            for nxt in self[transition.end][::-1]
                        )
            else:
                transitions.append((transition, cursor))

        return transitions

    def _match_suffix_no_backtrack(
        self, cursor: Cursor, context: Context
    ) -> Optional[Cursor]:
        # we only need to keep track of 3 state variables
        visited = set()
        queue = deque(
            self.queue_transition(
                Transition(EPSILON, self.start_state), cursor, context, visited
            )
        )

        match = None

        while True:
            frontier, visited = deque(), set()

            while queue:
                transition, cursor = queue.popleft()

                if transition.matcher is EPSILON or transition.matcher(cursor, context):
                    if transition.end in self.accepting_states:
                        match = transition.matcher.update(cursor)
                        break

                    frontier.extend(
                        self.queue_transition(transition, cursor, context, visited)
                    )

            if not frontier:
                break

            queue = frontier

        return match

    def _match_suffix_backtrack(
        self, cursor: Cursor, context: Context
    ) -> Optional[Cursor]:
        # we only need to keep track of 3 state variables
        stack = [(self.start_state, cursor, ())]

        while stack:
            state, cursor, path = stack.pop()  # type: (int, Cursor, tuple[int, ...])

            if state in self.accepting_states:
                return cursor

            for matcher, end_state in reversed(self.step(state, cursor, context)):
                if isinstance(matcher, Anchor):
                    if end_state in path:
                        continue
                    updated_path = path + (end_state,)
                else:
                    updated_path = ()

                stack.append(
                    (
                        end_state,
                        matcher.update(cursor),
                        updated_path,
                    )
                )

        return None

    def match_suffix(self, cursor: Cursor, context: Context) -> Optional[Cursor]:
        """
        Given a cursor, and context. Match the pattern against the cursor and return
        a final cursor that matches the pattern or none if the pattern could not match

        Parameters
        ----------
        cursor: Cursor
            An initial cursor object
        context: Context
            A static context object

        Returns
        -------
        Optional[Cursor]
            A cursor object in which cursor[0] is the position where the pattern ends in context.txt
            and cursor[1] are the filled out groups

        Examples
        --------
        >>> from sys import maxsize
        >>> pattern, text = '(ab)+', 'abab'
        >>> compiled_regex = RegexNFA(pattern)
        >>> ctx = Context(text, RegexFlag.NOFLAG)
        >>> start = 0
        >>> c = compiled_regex.match_suffix(Cursor(start, [maxsize, maxsize]), ctx)
        >>> c
        Cursor(position=4, groups=[2, 4])
        >>> end, groups = c
        >>> assert text[start: end] == 'abab'
        """
        if isinstance(super(), DFA):
            return self._match_suffix_dfa(self.start_state, cursor, context)
        elif RegexFlag.NO_BACKTRACK & self.parser.flags:
            return self._match_suffix_no_backtrack(cursor, context)
        else:
            return self._match_suffix_backtrack(cursor, context)

    def __repr__(self):
        return super().__repr__()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
