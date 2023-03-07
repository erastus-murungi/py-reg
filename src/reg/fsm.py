import functools
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import cache, reduce
from itertools import chain, count, product
from operator import itemgetter
from string import ascii_uppercase
from sys import maxsize
from typing import Any, Iterable, Optional, Union

import graphviz
from more_itertools import first_true, pairwise

from reg.matcher import Context, Cursor, RegexPattern
from reg.optimizer import Optimizer
from reg.parser import (
    EMPTY_STRING,
    EPSILON,
    GROUP_LINK,
    MATCH,
    Anchor,
    Expression,
    Group,
    Match,
    Matcher,
    RegexNode,
    RegexNodesVisitor,
    RegexParser,
)
from reg.utils import Fragment, RegexFlag

State = Union[int, str]


def init_counters():
    s = (
        "".join(t)
        for t in chain.from_iterable(
            (product(ascii_uppercase, repeat=i) for i in range(10))
        )
    )
    _ = next(s)  # ignoring the empty string
    return (
        count(0),
        s,
    )


counter, strcounter = init_counters()


def gen_state() -> int:
    return next(counter)


def gen_state_fragment() -> Fragment[State]:
    return Fragment(gen_state(), gen_state())


@dataclass(eq=True, slots=True)
class Transition:
    matcher: Matcher
    end: State

    def __iter__(self):
        yield from [self.matcher, self.end]

    def __hash__(self):
        return hash((self.matcher, self.end))

    def __lt__(self, other):
        return self.end < other.end


class NFA(
    defaultdict[State, list[Transition]],
    RegexNodesVisitor[Fragment[State]],
    RegexPattern,
):
    """Formally, an NFA is a 5-tuple (Q, Σ, q0, T, δ) where
        • Q is finite set of states;
        • Σ is alphabet of input symbols;
        • q0 is start state;
        • T is subset of Q giving the ``accept`` states;
        and
        • δ is the transition function.
    Now the transition function specifies a set of states rather than a state: it maps Q × Σ to { subsets of Q }.

    A backtracking NFA based regex pattern matcher

    Examples
    --------
    >>> pattern, text = '(ab)+', 'abab'
    >>> compiled_regex = NFA(pattern)
    >>> print(list(compiled_regex.finditer(text)))
    [RegexMatch(span=(0, 4), match='abab')]
    >>> print([m.groups() for m in compiled_regex.finditer(text)])
    [('ab',)]

    """

    __slots__ = ("symbols", "states", "accepting_states", "start_state")

    def __init__(
        self,
        pattern: Optional[str] = None,
        flags: RegexFlag = RegexFlag.OPTIMIZE,
        reduce_epsilons=True,
        group_count=0,
    ):
        super(NFA, self).__init__(list)
        self.alphabet: set[Matcher] = set()
        self.states: set[State] = set()
        if pattern is None:
            self.accepting_states = set()
            self.start_state = -1
            self._flags = flags
            self._group_count = group_count
        else:
            parser = RegexParser(pattern, flags)
            RegexPattern.__init__(self, parser.group_count, parser.flags)
            if RegexFlag.OPTIMIZE & parser.flags:
                Optimizer.run(parser.root)
            src, sink = parser.root.accept(self)
            accept_node = gen_state()
            self.add_transition(sink, accept_node, MATCH)
            self.start_state = src
            self.accepting_states = {accept_node}
            self.update_symbols_and_states()
            if reduce_epsilons:
                self.reduce_epsilons()

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
                transitions.extend(
                    filter(
                        lambda transition: transition.matcher(cursor, context),
                        self[state],
                    )
                )

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
                if (
                    transition.matcher is EPSILON  # consume all epsilons
                    or transition.matcher(cursor, context)
                ):
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

                if transition.matcher(cursor, context):
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
        >>> compiled_regex = NFA(pattern)
        >>> ctx = Context(text, RegexFlag.NOFLAG)
        >>> start = 0
        >>> c = compiled_regex.match_suffix(Cursor(start, (maxsize, maxsize)), ctx)
        >>> c
        Cursor(position=4, groups=(2, 4))
        >>> end, groups = c
        >>> assert text[start: end] == 'abab'
        """
        if RegexFlag.NO_BACKTRACK & self._flags:
            return self._match_suffix_no_backtrack(cursor, context)
        else:
            return self._match_suffix_backtrack(cursor, context)

    def update_symbols_and_states(self):
        self.states = set()
        for start in self:
            for matchable, end in self[start]:
                self.states.add(end)
                self.alphabet.add(matchable)
        self.alphabet -= {EPSILON, GROUP_LINK}

    def update_symbols(self):
        for transition in chain.from_iterable(self.values()):
            self.alphabet.add(transition.matcher)
        self.alphabet -= {EPSILON, GROUP_LINK}

    def transition(
        self, state: State, symbol: Matcher, _: bool = False
    ) -> tuple[State, ...]:
        return tuple(
            transition.end for transition in self[state] if transition.matcher == symbol
        )

    def add_transition(self, start: State, end: State, matcher: Matcher):
        self[start].append(Transition(matcher, end))

    def reverse_transitions(self, state: State):
        self[state].reverse()

    def epsilon(self, start: State, end: State):
        self.add_transition(start, end, EPSILON)

    def __repr__(self):
        return (
            f"FSM(states={tuple(sorted(self.states))}, "
            f"symbols={self.alphabet}, "
            f"start_state={self.start_state=}, "
            f"transitions={super().__repr__()}, "
            f"accept_states={self.accepting_states}) "
        )

    def n_transitions(self):
        return sum(len(transitions) for transitions in self.values())

    def to_json(self):
        class CustomEncoder(json.JSONEncoder):
            def default(self, o: Any) -> Any:
                if isinstance(o, Transition):
                    return [o.matcher, o.end]
                if isinstance(o, RegexNode):
                    return o.to_string()
                if isinstance(o, set):
                    return list(o)
                return json.JSONEncoder.default(self, o)

        return json.dumps(
            {
                "states": self.states,
                "transitions": self,
                "symbols": self.alphabet,
                "start_state": self.start_state,
                "accepting_states": self.accepting_states,
            },
            cls=CustomEncoder,
        )

    def subset_construction(
        self,
    ):
        seen, stack, finished = set(), [(self.start_state,)], []

        transitions = defaultdict(list)

        while stack:
            closure = self.epsilon_closure(stack.pop())
            # next we want to see which states are reachable from each of the states in the epsilon closure
            for matcher in self.alphabet:
                if move_closure := self.epsilon_closure(self.move(closure, matcher)):
                    transitions[closure].append((matcher, move_closure))
                    if move_closure not in seen:
                        seen.add(move_closure)
                        stack.append(move_closure)
            finished.append(closure)

        return transitions

    def epsilon_closure(self, states: Iterable[State]) -> tuple[State, ...]:
        """
        This is the set of all the nodes which can be reached by following epsilon labeled edges
        This is done here using a depth first search

        https://castle.eiu.edu/~mathcs/mat4885/index/Webview/examples/epsilon-closure.pdf
        """

        seen = set()
        stack = list(states)
        closure = set()

        while stack:
            if (state := stack.pop()) in seen:
                continue

            seen.add(state)
            # explore the states in the order which they are in
            stack.extend(self.transition(state, EPSILON, True)[::-1])
            closure.add(state)

        return tuple(sorted(closure))

    def move(self, states: Iterable[State], symbol: Matcher) -> frozenset[State]:
        return frozenset(
            reduce(
                set.union,
                (self.transition(state, symbol) for state in states),
                set(),
            )
        )

    def zero_or_more(self, fragment: Fragment[State], lazy: bool) -> Fragment[State]:
        empty_fragment = self.base(EMPTY_STRING)

        self.epsilon(fragment.end, empty_fragment.end)
        self.epsilon(fragment.end, fragment.start)
        self.epsilon(empty_fragment.start, fragment.start)

        if not lazy:
            self.reverse_transitions(empty_fragment.start)
            self.reverse_transitions(fragment.end)
        return empty_fragment

    def one_or_more(self, fragment: Fragment[State], lazy: bool) -> Fragment[State]:
        s = gen_state()

        self.epsilon(fragment.end, fragment.start)
        self.epsilon(fragment.end, s)

        if lazy:
            self.reverse_transitions(fragment.end)
        return Fragment(fragment.start, s)

    def zero_or_one(self, fragment: Fragment[State], lazy: bool) -> Fragment[State]:
        self.add_transition(fragment.start, fragment.end, EMPTY_STRING)
        if lazy:
            self.reverse_transitions(fragment.start)
        return fragment

    def alternation(
        self, lower: Fragment[State], upper: Fragment[State]
    ) -> Fragment[State]:
        fragment = gen_state_fragment()

        self.epsilon(fragment.start, lower.start)
        self.epsilon(fragment.start, upper.start)
        self.epsilon(lower.end, fragment.end)
        self.epsilon(upper.end, fragment.end)

        return fragment

    def concatenate(self, fragment1: Fragment[State], fragment2: Fragment[State]):
        self.epsilon(fragment1.end, fragment2.start)

    def base(
        self, matcher: Matcher, fragment: Optional[Fragment[State]] = None
    ) -> Fragment:
        if fragment is None:
            fragment = gen_state_fragment()
        self.add_transition(fragment.start, fragment.end, matcher)
        return fragment

    def graph(self):
        dot = graphviz.Digraph(
            self.__class__.__name__ + ", ".join(map(str, self.states)),
            format="pdf",
            engine="dot",
        )
        dot.attr("graph", rankdir="LR")
        dot.attr("node", fontname="verdana")
        dot.attr("edge", fontname="verdana")

        seen = set()

        for state, transitions in self.items():
            for transition in transitions:
                symbol, end = transition
                if state not in seen:
                    color = "green" if state == self.start_state else ""
                    dot.node(
                        str(state),
                        color=color,
                        shape="doublecircle"
                        if state in self.accepting_states
                        else "circle",
                        style="filled",
                    )
                    seen.add(state)
                if end not in seen:
                    dot.node(
                        f"{end}",
                        shape="doublecircle"
                        if end in self.accepting_states
                        else "circle",
                        style="filled",
                    )
                    seen.add(end)
                if symbol is GROUP_LINK:
                    dot.edge(str(state), str(end), color="blue", style="dotted")
                else:
                    dot.edge(str(state), str(end), label=str(symbol), color="black")

        dot.node("start", shape="none")
        dot.edge("start", f"{self.start_state}", arrowhead="vee")
        dot.render(view=True, directory="graphs", filename=str(id(self)))

    def _concat_fragments(self, fragments: Iterable[Fragment[[State]]]):
        for fragment1, fragment2 in pairwise(fragments):
            self.epsilon(fragment1.end, fragment2.start)

    def _apply_range_quantifier(self, node: Group | Match) -> Fragment[State]:
        """
        Generate a fragment for a group or match node with a range quantifier
        """

        quantifier = node.quantifier
        start, end = quantifier.param

        if start == 0:
            if end == maxsize:
                # 'a{0,} = a{0,maxsize}' expands to a*
                return self.zero_or_more(
                    self._gen_frag_for_quantifiable(node), quantifier.lazy
                )
            elif end is None:
                # a{0} = ''
                return self.base(EMPTY_STRING)

        if end is not None:
            if end == maxsize:
                # a{3,} expands to aaa+.
                # 'a{3,maxsize}
                fragments = [
                    self._gen_frag_for_quantifiable(node) for _ in range(start - 1)
                ] + [
                    self.one_or_more(
                        self._gen_frag_for_quantifiable(node), quantifier.lazy
                    )
                ]
            else:
                # a{,5} = a{0,5} or a{3,5}
                fragments = [self._gen_frag_for_quantifiable(node) for _ in range(end)]

                for fragment in fragments[start:end]:
                    # empty transitions all lead to a common exit
                    self.add_transition(fragment.start, fragments[-1].end, EMPTY_STRING)
                    if quantifier.lazy:
                        self.reverse_transitions(fragment.start)

        else:
            fragments = [self._gen_frag_for_quantifiable(node) for _ in range(start)]

        self._concat_fragments(fragments)
        return Fragment(fragments[0].start, fragments[-1].end)

    def _gen_frag_for_quantifiable(self, node: Group | Match) -> Fragment[State]:
        """
        Helper method to generate fragments for nodes and matches
        """
        if isinstance(node, Group):
            return self._add_capturing_markers(node.expression.accept(self), node)
        return node.item.accept(self)

    def _apply_quantifier(
        self,
        node: Group | Match,
    ) -> Fragment:
        quantifier = node.quantifier
        if isinstance(quantifier.param, str):
            fragment = self._gen_frag_for_quantifiable(node)
            match quantifier.param:
                case "+":
                    return self.one_or_more(fragment, quantifier.lazy)
                case "*":
                    return self.zero_or_more(fragment, quantifier.lazy)
                case "?":
                    return self.zero_or_one(fragment, quantifier.lazy)
                case _:
                    raise RuntimeError(f"unrecognized quantifier {quantifier.param}")
        else:
            return self._apply_range_quantifier(node)

    def _add_capturing_markers(
        self, fragment: Fragment[State], group: Group
    ) -> Fragment[State]:
        if group.is_capturing():
            markers_fragment = gen_state_fragment()
            self.base(
                Anchor.group_entry(group.index),
                Fragment(markers_fragment.start, fragment.start),
            )
            self.base(
                Anchor.group_exit(group.index),
                Fragment(fragment.end, markers_fragment.end),
            )
            self.add_transition(markers_fragment.end, fragment.start, GROUP_LINK)
            return markers_fragment
        return fragment

    def visit_expression(self, expression: Expression):
        fragments = [subexpression.accept(self) for subexpression in expression.seq]
        self._concat_fragments(fragments)
        fragment = Fragment(fragments[0].start, fragments[-1].end)
        if expression.alternate is None:
            return fragment
        return self.alternation(fragment, expression.alternate.accept(self))

    def visit_quantifiable(self, node: Group | Match) -> Fragment[State]:
        if node.quantifier:
            return self._apply_quantifier(node)
        return self._gen_frag_for_quantifiable(node)

    visit_group = visit_match = visit_quantifiable

    visit_character = (
        visit_character_group
    ) = (
        visit_any_character
    ) = visit_anchor = visit_word = lambda self, matcher: self.base(matcher)

    def _gen_frontier_transitions(self, start_state: State) -> list[Transition]:
        seen: set[State | Transition] = set()
        stack: list[State | Transition] = [start_state]
        frontier_transitions: list[Transition] = []

        while stack:
            if (item := stack.pop()) in seen:
                continue

            if isinstance(item, Transition):
                frontier_transitions.append(item)
                continue

            seen.add(item)

            next_in_stack = []
            for transition in self[item]:
                if transition.matcher is EPSILON:
                    next_in_stack.append(transition.end)
                elif transition not in frontier_transitions:
                    next_in_stack.append(transition)
            stack.extend(next_in_stack[::-1])

        return frontier_transitions

    def reduce_epsilons(self):
        """
        Attempts to reduce the number of epsilon's in the NFA while maintaining correctness
        """

        while True:
            queue = deque([self.start_state])
            visited = set()

            changed = False
            while queue:
                if (state := queue.pop()) in visited:
                    continue

                visited.add(state)

                transitions = []
                for transition in self[state]:
                    if transition.matcher is EPSILON:
                        # transitions of the form a -> ε -> b -> `matchable-s` -> `children` can be reduced to
                        # a -> `matchable-s` -> `children`
                        # special care is taken to ensure transitions order is maintained
                        for frontier_transition in self._gen_frontier_transitions(
                            transition.end
                        ):
                            if frontier_transition not in transitions:
                                transitions.append(frontier_transition)
                            queue.appendleft(frontier_transition.end)
                    else:
                        # can't take advantage of epsilon transition
                        if transition not in transitions:
                            transitions.append(transition)
                        queue.appendleft(transition.end)

                if transitions != self[state]:
                    changed = True

                self[state] = transitions

            self._prune_unreachable_transitions()
            self.update_symbols_and_states()

            if not changed:
                break

    def _prune_unreachable_transitions(self):
        seen = set()
        stack: list[State] = [self.start_state]
        reachable = set()

        while stack:
            if (state := stack.pop()) in seen:
                continue

            seen.add(state)

            stack.extend(transition.end for transition in self[state])
            reachable |= {(state, transition) for transition in self[state]}

        for state in self.states:
            for transition in self[state][:]:
                if (state, transition) not in reachable:
                    self[state].remove(transition)


class DFA(NFA):
    __slots__ = ("symbols", "states", "accepting_states", "start_state")

    def __init__(
        self,
        symbols,
        states,
        accepting_states,
        start_state,
        transitions,
        *,
        flags: RegexFlag = RegexFlag.NOFLAG,
        group_count=0,
    ):
        super().__init__(flags=flags, group_count=group_count)
        self.update(transitions)
        self.states = set(states)
        self.accepting_states = set(accepting_states)
        self.start_state = start_state
        self.symbols = symbols

    @staticmethod
    def from_pattern(pattern: str, flags: RegexFlag = RegexFlag.OPTIMIZE):
        nfa = NFA(pattern, flags)

        dfa_table, minimal_dfa_table = {}, defaultdict(list)
        dfa_accepting, minimal_dfa_accepting = [], []

        def state_generator(
            accepting_prev,
            accepting_curr,
            state_counter,
            sources,
        ):
            new_state = next(state_counter)
            if any(source in accepting_prev for source in sources):
                accepting_curr.append(new_state)
            return new_state

        def get_start_state(transition_table):
            reversed_transition_table = defaultdict(list)
            for start, transitions in transition_table.items():
                for _, end in transitions:
                    reversed_transition_table[end].append(start)

            return first_true(
                transition_table,
                pred=lambda state: not len(reversed_transition_table[state]),
            )

        gen_nfa_state, gen_dfa_state = cache(
            functools.partial(
                state_generator, nfa.accepting_states, dfa_accepting, counter
            )
        ), cache(
            functools.partial(
                state_generator, dfa_accepting, minimal_dfa_accepting, strcounter
            )
        )

        dfa_table = {
            gen_nfa_state(start_state): [
                (matcher, gen_nfa_state(end_state))
                for matcher, end_state in transitions
            ]
            for start_state, transitions in nfa.subset_construction().items()
        }

        partition_member2dfa_state = {
            partition_member: gen_dfa_state(partition)
            for partition in DFA.hopcroft(dfa_table, dfa_accepting, nfa.alphabet)
            for partition_member in partition
        }

        for start_state, transitions in dfa_table.items():
            dfa_state = partition_member2dfa_state[start_state]
            for matcher, end_state in transitions:
                if (
                    transition := Transition(
                        matcher, partition_member2dfa_state[end_state]
                    )
                ) not in minimal_dfa_table[dfa_state]:
                    minimal_dfa_table[dfa_state].append(transition)

        # dfa.graph()
        return DFA(
            nfa.alphabet,
            minimal_dfa_table.keys(),
            minimal_dfa_accepting,
            get_start_state(minimal_dfa_table),
            minimal_dfa_table,
            flags=nfa._flags,
            group_count=nfa._group_count,
        )

    @staticmethod
    def hopcroft(transitions, accepting, alphabet):
        partition: list[set[State]] = [
            set(transitions.keys()) - set(accepting),
            set(accepting),
        ]
        waiting: list[set[State]] = partition.copy()

        while waiting:
            s = waiting.pop()
            for c in alphabet:
                image = {
                    state
                    for state in transitions
                    if any(sym == c and end in s for sym, end in transitions[state])
                }
                for q in partition:
                    if (q1 := (image & q)) and (q2 := (q - q1)):
                        partition.extend((q1, q2))
                        partition.remove(q)

                        if q in waiting:
                            waiting.extend((q1, q2))
                            waiting.remove(q)
                        else:
                            waiting.append(min(q1, q2, key=len))
                        if s == q:
                            break
        return [tuple(sorted(p)) for p in partition]

    def _match_suffix_dfa(
        self, state: State, cursor: Cursor, context: Context, path: tuple[State, ...]
    ) -> Optional[int]:
        """
        This a fast matcher when you don't have groups or greedy quantifiers
        """
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
                if isinstance(matcher, Anchor):
                    if end_state in path:
                        continue
                    updated_path = path + (end_state,)
                else:
                    updated_path = ()
                result = self._match_suffix_dfa(
                    end_state, matcher.update_index(cursor), context, updated_path
                )

                if result is not None:
                    matching_cursors.append(result)

            if matching_cursors:
                return max(matching_cursors, key=itemgetter(0))

        return None

    def match_suffix(self, cursor: Cursor, context: Context) -> Optional[Cursor]:
        return self._match_suffix_dfa(
            self.start_state, cursor, context, (self.start_state,)
        )

    def clear(self) -> None:
        super().clear()
        self.alphabet.clear()
        self.start_state = -1
        self.accepting_states.clear()
        self.states.clear()


if __name__ == "__main__":
    # import doctest
    #
    # doctest.testmod()

    import re

    # regex, text = "(.*)c(.*)", "abcde"
    regex, text = ("([0a-z][a-z0-9]*,)+", "a5,b7,c9,")
    d = DFA.from_pattern(regex)
    print(list(d.finditer(text)))
    print(list(re.finditer(regex, text)))
