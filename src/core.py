import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import astuple, dataclass, field
from functools import reduce
from itertools import chain, combinations, count, product
from string import ascii_uppercase
from sys import maxsize
from typing import Any, Callable, Iterable, Iterator, Optional, Union

import graphviz
from more_itertools import first, first_true, minmax, pairwise, take

from .parser import (
    EMPTY_STRING,
    EPSILON,
    GROUP_LINK,
    Anchor,
    AnyCharacter,
    Character,
    CharacterGroup,
    Expression,
    Group,
    Match,
    Matchable,
    RegexNode,
    RegexNodesVisitor,
)
from .unionfind import UnionFind

State = Union[int, str]


def reset_state_gens():
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
        {},
    )


counter, strcounter, sourcescache = reset_state_gens()


def gen_state() -> int:
    return next(counter)


def gen_dfa_state(
    sources: Iterable[State],
    *,
    src_fsm: Optional["NFA"] = None,
    dst_fsm: Optional["NFA"] = None,
) -> str:
    sources = tuple(sorted(sources))
    if sources in sourcescache:
        return sourcescache[sources]

    strid = next(strcounter)
    if src_fsm is not None and dst_fsm is not None:
        if any(s == src_fsm.start_state for s in sources):
            dst_fsm.start_state = strid
        if any(s in src_fsm.accepting_states for s in sources):
            dst_fsm.accepting_states.add(strid)
    sourcescache[sources] = strid
    return strid


@dataclass(frozen=True, slots=True)
class Fragment:
    start: State = field(default_factory=gen_state)
    end: State = field(default_factory=gen_state)

    def __iter__(self):
        yield from astuple(self)


@dataclass(eq=True, slots=True)
class Transition:
    matchable: Matchable
    end: State

    def __iter__(self):
        yield from [self.matchable, self.end]


class NFA(defaultdict[State, list[Transition]], RegexNodesVisitor[Fragment]):
    """Formally, an NFA is a 5-tuple (Q, Σ, q0, T, δ) where
        • Q is finite set of states;
        • Σ is alphabet of input symbols;
        • q0 is start state;
        • T is subset of Q giving the ``accept`` states;
        and
        • δ is the transition function.
    Now the transition function specifies a set of states rather than a state: it maps Q × Σ to { subsets of Q }."""

    __slots__ = ("symbols", "states", "accepting_states", "start_state")

    def __init__(self):
        super(NFA, self).__init__(list)
        self.symbols: set[Matchable] = set()
        self.states: set[State] = set()
        self.accepting_states: set[State] = set()
        self.start_state: State = -1

    def update_symbols_and_states(self):
        for start, transitions in self.items():
            self.states.add(start)
            for symbol, end in transitions:
                self.states.add(end)
                self.symbols.add(symbol)
        self.symbols.discard(EPSILON)

    def update_symbols(self):
        for transition in chain.from_iterable(self.values()):
            self.symbols.add(transition.matchable)
        self.symbols.discard(EPSILON)

    def set_start(self, state: State):
        self.start_state = state

    def set_terminals(self, fragment: Fragment):
        self.set_start(fragment.start)
        self.set_accept(fragment.end)

    def set_accept(self, accept: State):
        self.accepting_states = {accept}

    def transition(
        self, state: State, symbol: Matchable, _: bool = False
    ) -> tuple[State, ...]:
        return tuple(
            transition.end
            for transition in self[state]
            if transition.matchable == symbol
        )

    def add_transition(self, start: State, end: State, matchable: Matchable):
        self[start].append(Transition(matchable, end))

    def reverse_transitions(self, state: State):
        self[state].reverse()

    def epsilon(self, start: State, end: State):
        self.add_transition(start, end, EPSILON)

    def __repr__(self):
        return (
            f"FSM({self.states=}, "
            f"{self.symbols=}, "
            f"{self.start_state=}, "
            f"transitions = {super().__repr__()}"
            f"accept_states={self.accepting_states}) "
        )

    def to_json(self):
        class CustomEncoder(json.JSONEncoder):
            def default(self, o: Any) -> Any:
                if isinstance(o, Transition):
                    return [o.matchable, o.end]
                if isinstance(o, RegexNode):
                    return o.string()
                if isinstance(o, set):
                    return list(o)
                return json.JSONEncoder.default(self, o)

        return json.dumps(
            {
                "states": self.states,
                "transitions": self,
                "symbols": self.symbols,
                "start_state": self.start_state,
                "accepting_states": self.accepting_states,
            },
            cls=CustomEncoder,
        )

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
            nxt = self.transition(state, EPSILON, True)[::-1]
            stack.extend(nxt)
            closure.add(state)

        return tuple(closure)

    def move(self, states: Iterable[State], symbol: Matchable) -> frozenset[State]:
        return frozenset(
            reduce(
                set.union,
                (self.transition(state, symbol) for state in states),
                set(),
            )
        )

    def zero_or_more(self, fragment: Fragment, lazy: bool) -> Fragment:
        empty_fragment = self.base(EMPTY_STRING)

        self.epsilon(fragment.end, empty_fragment.end)
        self.epsilon(fragment.end, fragment.start)
        self.epsilon(empty_fragment.start, fragment.start)

        if not lazy:
            self.reverse_transitions(empty_fragment.start)
            self.reverse_transitions(fragment.end)
        return empty_fragment

    def one_or_more(self, fragment: Fragment, lazy: bool) -> Fragment:
        s = gen_state()

        self.epsilon(fragment.end, fragment.start)
        self.epsilon(fragment.end, s)

        if lazy:
            self.reverse_transitions(fragment.end)
        return Fragment(fragment.start, s)

    def zero_or_one(self, fragment: Fragment, lazy: bool) -> Fragment:
        self.add_transition(fragment.start, fragment.end, EMPTY_STRING)
        if lazy:
            self.reverse_transitions(fragment.start)
        return fragment

    def alternation(self, lower: Fragment, upper: Fragment) -> Fragment:
        fragment = Fragment()

        self.epsilon(fragment.start, lower.start)
        self.epsilon(fragment.start, upper.start)
        self.epsilon(lower.end, fragment.end)
        self.epsilon(upper.end, fragment.end)

        return fragment

    def concatenate(self, fragment1: Fragment, fragment2: Fragment):
        self.epsilon(fragment1.end, fragment2.start)

    def base(
        self, matchable: Matchable, fragment: Optional[Fragment] = None
    ) -> Fragment:
        if fragment is None:
            fragment = Fragment()
        self.add_transition(fragment.start, fragment.end, matchable)
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
                    )
                    seen.add(end)
                if symbol is GROUP_LINK:
                    dot.edge(str(state), str(end), color="blue", style="dotted")
                else:
                    dot.edge(str(state), str(end), label=str(symbol), color="black")

        dot.node("start", shape="none")
        dot.edge("start", f"{self.start_state}", arrowhead="vee")
        dot.render(view=True, directory="graphs", filename=str(id(self)))

    def _concat_fragments(self, fragments: Iterable[Fragment]):
        for fragment1, fragment2 in pairwise(fragments):
            self.epsilon(fragment1.end, fragment2.start)

    def _apply_range_quantifier(self, node: Group | Match) -> Fragment:
        """
        Generate a fragment for a group or match node with a range quantifier
        """

        quantifier = node.quantifier
        start, end = quantifier.range_quantifier

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

    def _gen_frag_for_quantifiable(self, node: Group | Match) -> Fragment:
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
        if quantifier.char is not None:
            fragment = self._gen_frag_for_quantifiable(node)
            match quantifier.char:
                case "+":
                    return self.one_or_more(fragment, quantifier.lazy)
                case "*":
                    return self.zero_or_more(fragment, quantifier.lazy)
                case "?":
                    return self.zero_or_one(fragment, quantifier.lazy)
                case _:
                    raise RuntimeError(f"unrecognized quantifier {quantifier.char}")
        else:
            return self._apply_range_quantifier(node)

    def _add_capturing_markers(self, fragment: Fragment, group: Group) -> Fragment:
        if group.is_capturing():
            markers_fragment = Fragment()
            self.base(
                Anchor.group_entry(group.group_index),
                Fragment(markers_fragment.start, fragment.start),
            )
            self.base(
                Anchor.group_exit(group.group_index),
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

    def visit_group(self, group: Group):
        if group.quantifier:
            return self._apply_quantifier(group)
        return self._add_capturing_markers(group.expression.accept(self), group)

    def visit_match(self, match: Match) -> Fragment:
        if match.quantifier:
            return self._apply_quantifier(match)
        return match.item.accept(self)

    def visit_anchor(self, anchor: Anchor):
        return self.base(anchor)

    def visit_any_character(self, any_character: AnyCharacter) -> Fragment:
        return self.base(any_character)

    def visit_character(self, character: Character) -> Fragment:
        return self.base(character)

    def visit_character_group(self, character_group: CharacterGroup) -> Fragment:
        return self.base(character_group)


class DFA(NFA):
    __slots__ = ("symbols", "states", "accepting_states", "start_state")

    def __init__(self, nfa: Optional[NFA] = None):
        super(DFA, self).__init__()
        if nfa is not None:
            self._subset_construction(nfa)

    def transition(
        self, state: State, symbol: Matchable, wrapped: bool = False
    ) -> State | tuple[State, ...]:
        result = first_true(
            self[state],
            None,
            lambda transition: transition.matchable == symbol,
        )
        if result is None:
            return () if wrapped else ""
        return (result.end,) if wrapped else result.end

    def copy(self):
        cp = DFA()
        for state, transitions in self.items():
            cp[state] = transitions.copy()
        cp.symbols = self.symbols.copy()
        cp.start_state = self.start_state
        cp.accepting_states = self.accepting_states.copy()
        cp.states = self.states.copy()
        return cp

    def _subset_construction(self, nfa: NFA):
        seen, work_list, finished = set(), [(nfa.start_state,)], []

        while work_list:
            closure = nfa.epsilon_closure(work_list.pop())
            dfa_state = gen_dfa_state(closure, src_fsm=nfa, dst_fsm=self)
            # next we want to see which states are reachable from each of the states in the epsilon closure
            for symbol in nfa.symbols:
                if move_closure := nfa.epsilon_closure(nfa.move(closure, symbol)):
                    end_state = gen_dfa_state(move_closure, src_fsm=nfa, dst_fsm=self)
                    self.add_transition(dfa_state, end_state, symbol)
                    if move_closure not in seen:
                        seen.add(move_closure)
                        work_list.append(move_closure)
            finished.append(dfa_state)

        self.states.update(finished)
        self.update_symbols()
        self.start = first(finished)

    def gen_equivalence_states(self) -> Iterator[set[State]]:
        """
        Myhill-Nerode Theorem
        https://www.cs.scranton.edu/~mccloske/courses/cmps364/dfa_minimize.html
        """

        # a state is indistinguishable from itself
        indistinguishable = {(p, p) for p in self.states}

        for p, q in combinations(self.states, 2):
            # a pair of states are maybe indistinguishable
            # if they are both accepting or both non-accepting
            # we use min max to provide an ordering based on the labels
            p, q = minmax(p, q)
            if (p in self.accepting_states) == (q in self.accepting_states):
                indistinguishable.add((p, q))

        union_find = UnionFind(self.states)

        changed = True
        while changed:
            changed = False
            removed = set()
            for p, q in indistinguishable:
                if p == q:
                    continue
                # if two states are maybe indistinguishable, then do some more work to prove they are actually
                # indistinguishable
                for a in self.symbols:
                    km = minmax(self.transition(p, a), self.transition(q, a))
                    if km != ("", "") and km not in indistinguishable:
                        removed.add((p, q))
                        changed = True
            indistinguishable = indistinguishable - removed

        for p, q in indistinguishable:
            union_find.union(p, q)

        return union_find.to_sets()

    def minimize(self):
        accept_states = self.accepting_states.copy()
        states = set()
        lost = {}

        for sources in self.gen_equivalence_states():
            compound = gen_dfa_state(sources, src_fsm=self, dst_fsm=self)
            states.add(compound)
            for original in sources:
                lost[original] = compound

        self.states = states
        self.accepting_states = self.accepting_states - accept_states

        for start in tuple(self):
            if start in lost:
                self[lost.get(start)] = self.pop(start)

        for start in self:
            new_transitions = []
            for transition in tuple(self[start]):
                if transition.end in lost:

                    new_transitions.append(
                        Transition(transition.matchable, lost.get(transition.end))
                    )
                else:
                    new_transitions.append(transition)
            self[start] = new_transitions

    def clear(self) -> None:
        super().clear()
        self.symbols.clear()
        self.start_state = -1
        self.accepting_states.clear()
        self.states.clear()


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


class RegexPattern(ABC):
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

    def match(self, text: str):
        """Try to apply the pattern at the start of the string, returning
        a Match object, or None if no match was found."""
        return first_true(self.finditer(text), default=None)

    def findall(self, text):
        return [m.group(0) for m in self.finditer(text)]

    @abstractmethod
    def _match_at_index(self, text: str, index: int) -> Optional[MatchResult]:
        ...

    def _sub(
        self, string: str, replacer: str | Callable[[Match], str], count: int = maxsize
    ) -> tuple[str, int]:
        if isinstance(replacer, str):

            def r(_):
                return replacer

        else:
            r = replacer
        matches = take(count, self.finditer(string))
        chunks = []
        start = 0
        subs = 0
        for match in matches:
            chunks.append(string[start : match.start])
            chunks.append(r(match))
            start = match.end
            subs += 1
        chunks.append(string[start:])
        return "".join(chunks), subs

    def subn(
        self, string: str, replacer: str | Callable[[Match], str], count: int = maxsize
    ) -> tuple[str, int]:
        return self._sub(string, replacer, count)

    def sub(
        self, string: str, replacer: str | Callable[[Match], str], count: int = maxsize
    ):
        return self._sub(string, replacer, count)[0]
