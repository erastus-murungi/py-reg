from abc import ABC, abstractmethod
from collections import defaultdict
from copy import copy
from dataclasses import astuple, dataclass, field
from enum import Enum, IntFlag, auto
from functools import cache, reduce
from itertools import chain, combinations, count, product
from math import inf
from string import ascii_uppercase
from sys import maxsize
from typing import ClassVar, Collection, Final, Iterable, Iterator, Optional, Union

import graphviz
from more_itertools import minmax, pairwise

from unionfind import UnionFind


class RegexFlag(IntFlag):
    NOFLAG = auto()
    IGNORECASE = auto()
    MULTILINE = auto()
    DOTALL = auto()  # make dot match newline


class Matchable(ABC):
    @abstractmethod
    def match(self, text: str, position: int, flags: RegexFlag) -> bool:
        ...


def yield_letters():
    it = map(
        lambda t: "".join(t),
        chain.from_iterable((product(ascii_uppercase, repeat=i) for i in range(10))),
    )
    _ = next(it)
    yield from it


class State:
    ids: ClassVar = count(-1)

    def __init__(self, *, is_start=False, accepts=False, lazy=False):
        self.is_start = is_start
        self.accepts = accepts
        self.id = self._gen_id()
        self.lazy = lazy

    def _gen_id(self):
        return next(self.ids)

    def __repr__(self):
        return f"{self.id}"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id

    def is_null(self):
        return self is NullState


NullState: Final[State] = State(is_start=False, accepts=False)


class DFAState(State):
    _labels_gen = yield_letters()

    def __init__(self, from_states, *, is_start=False, accepts=False):
        self.sources = from_states
        super().__init__(is_start=is_start, accepts=accepts)

    def _gen_id(self):
        return self._gen_label(self.sources)

    @staticmethod
    @cache
    def _gen_label(_frozen):
        return next(DFAState._labels_gen)

    def is_null(self):
        return self is NullDfaState


NullDfaState: Final[State] = DFAState(from_states=None)


@dataclass(frozen=True)
class Fragment:
    start: State = field(default_factory=State)
    end: State = field(default_factory=State)

    def __iter__(self):
        yield from astuple(self)


@dataclass(frozen=True)
class Transition(Matchable):
    matchable: Matchable
    end: State

    def match(
        self,
        text: str,
        position: int,
        flags: RegexFlag,
        default: Optional[State] = None,
    ) -> Optional[State]:
        if self.matchable.match(text, position, flags):
            return self.end
        return default

    def __iter__(self):
        # noinspection PyRedundantParentheses
        yield from (self.matchable, self.end)


class FiniteStateAutomaton(
    defaultdict[State, set[Transition]],
    ABC,
):
    states: set[State]
    start_state: State
    accept: State | set[DFAState]
    symbols: set[Matchable]

    @abstractmethod
    def transition(
        self, state: State, symbol: Matchable
    ) -> DFAState | Collection[State]:
        pass

    @abstractmethod
    def all_transitions(self) -> tuple[Matchable, State, State]:
        pass

    def graph(self):
        dot = graphviz.Digraph(
            self.__class__.__name__ + ", ".join(map(str, self.states)),
            format="pdf",
            engine="circo",
        )
        dot.attr("graph", rankdir="LR")

        seen = set()

        for symbol, start, end in self.all_transitions():
            if start not in seen:
                if start.is_start:
                    dot.node(
                        str(start.id),
                        color="green",
                        shape="doublecircle" if start.accepts else "circle",
                        style="filled",
                    )
                elif start.lazy:
                    dot.node(
                        str(start.id),
                        color="red",
                        shape="doublecircle" if start.accepts else "circle",
                        style="filled",
                    )
                else:
                    dot.node(
                        f"{start.id}",
                        shape="doublecircle" if start.accepts else "circle",
                    )
                seen.add(start)
            if end not in seen:
                seen.add(end)
                if end.lazy:
                    dot.node(
                        f"{end.id}",
                        color="gray",
                        style="filled",
                        shape="doublecircle" if end.accepts else "circle",
                    )
                else:
                    dot.node(
                        f"{end.id}", shape="doublecircle" if end.accepts else "circle"
                    )
            dot.edge(str(start.id), str(end.id), label=str(symbol))

        dot.node("start", shape="none")
        dot.edge("start", f"{self.start_state.id}", arrowhead="vee")
        dot.render(view=True, directory="graphs", filename=str(id(self)))

    def update_symbols_and_states(self):
        for symbol, start, end in self.all_transitions():
            self.states.update({start, end})
            self.symbols.add(symbol)
        self.symbols.discard(Epsilon)

    def _dict(self) -> defaultdict[State, set[Transition]]:
        d = defaultdict(set)
        for state, transitions in self.items():
            d[state] = transitions.copy()
        return d

    def set_start(self, state: State):
        self.start_state = state
        self.start_state.is_start = True

    @abstractmethod
    def create_state(self, sources) -> DFAState:
        pass

    @abstractmethod
    def create_transition(self, start: State, end: State, matchable: Matchable) -> None:
        pass


class NFA(FiniteStateAutomaton):
    """Formally, an NFA is a 5-tuple (Q, Σ, q0, T, δ) where
        • Q is finite set of states;
        • Σ is alphabet of input symbols;
        • q0 is start state;
        • T is subset of Q giving the ``accept`` states;
        and
        • δ is the transition function.
    Now the transition function specifies a set of states rather than a state: it maps Q × Σ to { subsets of Q }."""

    def __init__(self):
        super(FiniteStateAutomaton, self).__init__(set)
        self.symbols = set()
        self.states = set()

    def set_accept(self, accept: State):
        self.accept = accept
        accept.accepts = True

    def all_transitions(self):
        for start, transitions in self.items():
            for symbol, end in transitions:
                yield symbol, start, end

    def transition(self, state: State, symbol: Matchable) -> list[State]:
        states = []
        for sym, state in self[state]:
            if sym == symbol:
                states.append(state)
        if not states:
            states = [NullState]
        return states

    def create_transition(self, start: State, end: State, matchable: Matchable):
        self[start].add(Transition(matchable, end))

    def epsilon(self, start: State, end: State):
        self.create_transition(start, end, Epsilon)

    def __repr__(self):
        return (
            f"FSM(states={self.states}, "
            f"symbols={self.symbols}, "
            f"start_state={self.start_state}, "
            f"accept_states={self.accept}) "
        )

    def filter(self, start: State, matchable: Matchable) -> tuple[Transition, ...]:
        return tuple(state for symbol, state in self[start] if symbol == matchable)

    def epsilon_closure(self, states: Iterable[State]) -> frozenset:
        """
        This is the set of all the nodes which can be reached by following epsilon labeled edges
        This is done here using a depth first search

        https://castle.eiu.edu/~mathcs/mat4885/index/Webview/examples/epsilon-closure.pdf
        """

        seen = set()

        stack = list(states)
        closure = set()

        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)

            stack.extend(self.filter(u, Epsilon))
            closure.add(u)

        return frozenset(closure)

    def move(self, states: Iterable[State], symbol: Matchable) -> frozenset[State]:
        return frozenset(
            reduce(set.union, (self.filter(state, symbol) for state in states), set())
        )

    def find_state(self, state_id: int) -> Optional[State]:
        for state in self.states:
            if state_id == state.id:
                return state
        return None

    def create_state(self, sources) -> DFAState:
        state = DFAState(from_states=sources)
        if self.accept in state.sources:
            state.accepts = True
        state.lazy = any(s.lazy for s in sources)
        return state

    def zero_or_more(self, fragment: Fragment, lazy: bool) -> Fragment:
        new_fragment = Fragment()

        self.epsilon(fragment.end, fragment.start)
        self.epsilon(new_fragment.start, fragment.start)
        self.epsilon(fragment.end, new_fragment.end)
        self.epsilon(new_fragment.start, new_fragment.end)

        empty_fragment = self.base(Anchor.empty_string())
        self.concatenate(empty_fragment, new_fragment)
        fragment.start.lazy = fragment.end.lazy = lazy
        return new_fragment

    def one_or_more(self, fragment: Fragment, lazy: bool) -> Fragment:
        source, sink = fragment
        self.epsilon(sink, source)
        source.lazy = sink.lazy = lazy
        return fragment

    def zero_or_one(self, fragment: Fragment, lazy: bool) -> Fragment:
        self.epsilon(fragment.start, fragment.end)
        base_fragment = self.base(Anchor.empty_string())
        self.concatenate(base_fragment, fragment)
        fragment.start.lazy = fragment.end.lazy = lazy
        return Fragment(base_fragment.start, fragment.end)

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
        self,
        matchable: Matchable,
    ) -> Fragment:
        fragment = Fragment()
        self.create_transition(fragment.start, fragment.end, matchable)
        return fragment


class DFA(FiniteStateAutomaton):
    def __init__(self, nfa: Optional[NFA] = None):
        super(FiniteStateAutomaton, self).__init__(set)

        self.states: set[DFAState] = set()
        self.symbols: set[Matchable] = set()
        self.accept: set[DFAState] = set()
        if nfa is not None:
            self.subset_construction(nfa)

    def transition(self, state: State, symbol: Matchable) -> State:
        for sym, state in self[state]:
            if sym == symbol:
                return state
        return NullDfaState

    def subset_construction(self, nfa: NFA):
        s0 = DFAState(from_states=frozenset({nfa.start_state}))
        seen, stack = set(), []

        def compute_transitions_for_dfa_state(
            state: DFAState,
        ):
            # what is the epsilon closure of the dfa_states
            closure_items = nfa.epsilon_closure(state.sources)
            d = nfa.create_state(closure_items)
            if d.accepts:
                self.accept.add(d)

            # next we want to see which states are reachable from each of the states in the epsilon closure
            for symbol in nfa.symbols:
                next_states_set = nfa.epsilon_closure(nfa.move(closure_items, symbol))
                # new DFAState
                df = nfa.create_state(next_states_set)
                self.create_transition(d, df, symbol)
                if next_states_set not in seen:
                    seen.add(next_states_set)
                    stack.append(df)
            return d

        self.set_start(compute_transitions_for_dfa_state(s0))

        while stack:
            compute_transitions_for_dfa_state(stack.pop())

        self.clean_up_empty_sets()
        self.update_symbols_and_states()
        self.symbols = nfa.symbols

    def clean_up_empty_sets(self):
        items = self._dict().items()
        self.clear()
        for state, transitions in items:
            for symbol, end in transitions:
                if end.sources:
                    self.create_transition(state, end, symbol)

    def all_transitions(self):
        for state, transitions in self.items():
            for symbol, end in transitions:
                yield symbol, state, end

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
            if p.accepts == q.accepts:
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
                    if (
                        km != (NullDfaState, NullDfaState)
                        and km not in indistinguishable
                    ):
                        removed.add((p, q))
                        changed = True
            indistinguishable = indistinguishable - removed

        for p, q in indistinguishable:
            union_find.union(p, q)

        return union_find.to_sets()

    def create_state(self, sources: set[State]):
        if len(sources) == 1:
            return sources.pop()
        state = DFAState(from_states=frozenset(sources))
        if self.start_state in state.sources:
            state.is_start = True
        for accept_state in self.accept:
            if accept_state in state.sources:
                state.accepts = True
                break
        state.lazy = any(s.lazy for s in sources)
        return state

    def minimize(self):
        self.states: set[DFAState] = set(
            map(self.create_state, self.gen_equivalence_states())
        )

        lost = {
            original: compound
            for compound in self.states
            for original in compound.sources
            if len(compound.sources) > 1
        }

        for a in list(self.keys()):
            if a in lost:
                self[lost.get(a)] = self.pop(a)
        for a in self:
            for symbol, b in self[a].copy():
                if b in lost:
                    self.create_transition(a, lost.get(b), symbol)
        (start_state,) = tuple(filter(lambda s: s.is_start, self.states))
        self.set_start(start_state)
        self.accept = set(filter(lambda s: s.accepts, self.accept))

    def create_transition(self, start: State, end: State, matchable: Matchable) -> None:
        self[start].add(Transition(matchable, end))


ESCAPED = set(". \\ + * ? [ ^ ] $ ( ) { } = ! < > | -".split())

anchors = {"$", "^"}

character_classes = {"w", "W", "s", "S", "d", "D"}

no_escape_in_group = {"\\", "-", "[", ":", ".", ">", ">"}


@dataclass
class RegexNode(ABC):
    pos: int = field(repr=False)

    @abstractmethod
    def fsm(self, nfa: NFA) -> Fragment:
        ...


class Operator(ABC):
    pass


class QuantifierItem(ABC):
    pass


class QuantifierType(Enum):
    OneOrMore = "+"
    ZeroOrMore = "*"
    ZeroOrOne = "?"

    @staticmethod
    def get(char):
        match char:
            case "+":
                return QuantifierType.OneOrMore
            case "*":
                return QuantifierType.ZeroOrMore
            case "?":
                return QuantifierType.ZeroOrOne
            case _:
                raise ValueError(f"unrecognized quantifier {char}")


@dataclass
class QuantifierChar(QuantifierItem):
    type: QuantifierType


@dataclass
class Quantifier(Operator):
    item: QuantifierChar
    lazy: bool = False

    def apply(self, fragment: Fragment, nfa: NFA) -> Fragment:
        match self.item.type:
            case QuantifierType.OneOrMore:
                return nfa.one_or_more(fragment, self.lazy)
            case QuantifierType.ZeroOrMore:
                return nfa.zero_or_more(fragment, self.lazy)
            case QuantifierType.ZeroOrOne:
                return nfa.zero_or_one(fragment, self.lazy)
            case _:
                raise NotImplementedError


@dataclass
class RangeQuantifier(QuantifierItem):
    start: Optional[int]
    end: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.start, int) and self.end is None and self.start < 1:
            raise ValueError(f"fixed quantifier, {{n}} must be >= 1: not {self.start}")
        elif isinstance(self.start, int) and isinstance(self.end, int):
            if self.start < 0:
                raise ValueError(
                    f"for {{n, m}} quantifier, {{n}} must be >= 0: not {self.start}"
                )
            if self.end < self.start:
                raise ValueError(
                    f"for {{n, m}} quantifier, {{m}} must be >= {{n}}: not {self.end}"
                )
        elif isinstance(self.start, int) and self.end == inf:
            if self.start < 0:
                raise ValueError(
                    f"for {{n,}} quantifier, {{n}} must be >= 0: not {self.start}"
                )
        elif self.start == 0:
            if not isinstance(self.end, int):
                raise ValueError(f"invalid upper bound {self.end}")
            if self.end < 1:
                raise ValueError(
                    f"for {{, m}} quantifier, {{m}} must be >= 1: not {self.end}"
                )
        else:
            raise ValueError(f"invalid range {{{self.start}, {self.end}}}")

    def expand(self, item: Union["SubExpressionItem", "Expression"], lazy: bool):
        # e{3} expands to eee; e{3,5} expands to eeee?e?, and e{3,} expands to eee+.

        seq = [copy(item) for _ in range(self.start)]

        if self.end is not None:
            if self.end == inf:
                if self.start > 0:
                    item = seq.pop()
                    seq.append(
                        Group(
                            item.pos,
                            Expression(item.pos, [item]),
                            Quantifier(QuantifierChar(QuantifierType.OneOrMore), lazy),
                        )
                    )
                else:
                    seq.append(
                        Group(
                            item.pos,
                            Expression(item.pos, [item]),
                            Quantifier(QuantifierChar(QuantifierType.ZeroOrMore), lazy),
                        )
                    )

            else:
                for _ in range(self.start, self.end):
                    seq.append(
                        Group(
                            item.pos,
                            Expression(item.pos, [item]),
                            Quantifier(QuantifierChar(QuantifierType.ZeroOrOne), lazy),
                        )
                    )
        return Expression(item.pos, seq)


class SubExpressionItem(RegexNode, ABC):
    pass


@dataclass
class Expression(RegexNode):
    seq: list[SubExpressionItem]
    alternate: Optional["Expression"] = None

    def fsm(self, nfa: NFA) -> Fragment:
        fragments = [subexpression.fsm(nfa) for subexpression in self.seq]
        for fragment1, fragment2 in pairwise(fragments):
            nfa.epsilon(fragment1.end, fragment2.start)
        fragment = Fragment(fragments[0].start, fragments[-1].end)
        if self.alternate is None:
            return fragment
        return nfa.alternation(fragment, self.alternate.fsm(nfa))


@dataclass
class Group(SubExpressionItem):
    expression: Expression
    quantifier: Optional[Quantifier]
    capturing: bool = False

    def fsm(self, nfa: NFA) -> Fragment:
        fragment = self.expression.fsm(nfa)
        if self.quantifier:
            return self.quantifier.apply(fragment, nfa)
        return fragment


class MatchItem(SubExpressionItem, ABC):
    pass


@dataclass
class Match(SubExpressionItem):
    item: MatchItem
    quantifier: Optional[Quantifier]

    def fsm(self, nfa: NFA) -> Fragment:
        fragment = self.item.fsm(nfa)
        if self.quantifier is None:
            return fragment
        return self.quantifier.apply(fragment, nfa)


@dataclass
class MatchAnyCharacter(MatchItem, Matchable):
    ignore: tuple = ("ε", "\n")

    def fsm(self, nfa: NFA) -> Fragment:
        return nfa.base(self)

    def __eq__(self, other):
        return isinstance(other, MatchAnyCharacter) and other.ignore == self.ignore

    def match(self, text, position, flags) -> bool:
        return position < len(text) and text[position] not in self.ignore

    def __repr__(self):
        return "Any"

    def __hash__(self):
        return hash(".")


class CharacterGroupItem(Matchable, ABC):
    pass


@dataclass
class CharacterScalar(CharacterGroupItem, MatchItem):
    char: str

    def fsm(self, nfa: NFA) -> Fragment:
        fragment = Fragment()
        nfa.create_transition(fragment.start, fragment.end, self)
        return fragment

    def match(self, text, position, flags) -> bool:
        if position < len(text):
            return self.char == text[position]
        return False

    def __eq__(self, other) -> bool:
        return other == self.char

    def __lt__(self, other) -> bool:
        if isinstance(other, CharacterScalar):
            return self.char <= other.char
        return other <= self.char

    def __repr__(self):
        return f"{self.char}"

    def __hash__(self):
        return hash(self.char)


Epsilon = CharacterScalar(-maxsize, "ε")


class MatchCharacterClass(MatchItem, ABC):
    pass


@dataclass
class CharacterGroup(MatchCharacterClass, Matchable):
    items: tuple[CharacterGroupItem, ...]
    negated: bool = False

    def fsm(self, nfa: NFA) -> Fragment:
        fragment = Fragment()
        nfa.create_transition(fragment.start, fragment.end, self)
        return fragment

    def match(self, text, position, flags) -> bool:
        if position >= len(text):
            return False
        return self.negated ^ any(
            item.match(text, position, flags) for item in self.items
        )

    def __eq__(self, other):
        if isinstance(other, CharacterGroup):
            return self.items == other.items
        return False

    def __repr__(self):
        return f"[{('^' if self.negated else '')}{', '.join(map(repr, self.items))}]"

    def __lt__(self, other):
        return id(self) < id(other)

    def __hash__(self):
        return hash((self.items, self.negated))


@dataclass
class CharacterRange(CharacterGroupItem, Matchable):
    start: str
    end: str

    def match(self, text, position, flags) -> bool:
        if position < len(text):
            return self.start <= text[position] <= self.end
        return False

    def __post_init__(self):
        if self.start > self.end:
            raise ValueError(f"[{self.start}-{self.end}] is not ordered")

    def __hash__(self):
        return hash((self.start, self.end))

    def __repr__(self):
        return f"[{self.start}-{self.end}]"


class AnchorType(Enum):
    StartOfString = "^"
    EndOfString = "$"
    EmptyString = "nothing to see here"

    # must be escaped
    WordBoundary = "b"
    NonWordBoundary = "B"
    AnchorStartOfStringOnly = "A"
    AnchorEndOfStringOnlyNotNewline = "z"
    AnchorEndOfStringOnly = "Z"
    AnchorPreviousMatchEnd = "G"

    @staticmethod
    def get(char):
        match char:
            case "^":
                return AnchorType.StartOfString
            case "$":
                return AnchorType.EndOfString
            case "b":
                return AnchorType.WordBoundary
            case "B":
                return AnchorType.NonWordBoundary
            case "A":
                return AnchorType.AnchorStartOfStringOnly
            case "z":
                return AnchorType.AnchorEndOfStringOnlyNotNewline
            case "Z":
                return AnchorType.AnchorEndOfStringOnly
            case "G":
                return AnchorType.AnchorPreviousMatchEnd
            case _:
                raise ValueError(f"unrecognized anchor {char}")


def is_word_character(char: str) -> bool:
    return len(char) == 1 and char.isalpha() or char == "_"


def is_word_boundary(text: str, position: int) -> bool:
    # There are three different positions that qualify as word boundaries:
    #
    # 1. Before the first character in the string, if the first character is a word character.
    # 2. After the last character in the string, if the last character is a word character.
    # 3. Between two characters in the string,
    #           where one is a word character and the other is not a word character.
    case1 = len(text) > 0 and position == 0 and is_word_character(text[position])
    case2 = (1 <= len(text) <= position and is_word_character(text[position - 1])) or (
        len(text) >= 2
        and position == len(text) - 1
        and text[position] == "\n"
        and is_word_character(text[position - 2])
    )
    case3 = (position - 1 >= 0 and position < len(text)) and (
        (
            not is_word_character(text[position - 1])
            and is_word_character(text[position])
        )
        or (
            is_word_character(text[position - 1])
            and not is_word_character(text[position])
        )
    )
    return case1 or case2 or case3


@dataclass(slots=True)
class Anchor(SubExpressionItem, Matchable):
    anchor_type: AnchorType

    def fsm(self, nfa: NFA) -> Fragment:
        return nfa.base(self)

    @staticmethod
    def empty_string(pos: int = maxsize) -> "Anchor":
        return Anchor(pos, AnchorType.EmptyString)

    def match(self, text, position, flags) -> bool:
        match self.anchor_type:
            case AnchorType.StartOfString:
                # assert that this is the beginning of the string
                return (
                    position == 0
                )  # or the previous char is a \n if MULTILINE mode enabled
            case AnchorType.EndOfString:
                return (
                    position >= len(text)
                    or position == len(text) - 1
                    and text[position] == "\n"
                )
            case AnchorType.WordBoundary:
                return is_word_boundary(text, position)
            case AnchorType.NonWordBoundary:
                return text and not is_word_boundary(text, position)
            case AnchorType.EmptyString:
                return True

        raise NotImplementedError

    def __hash__(self):
        return hash(self.anchor_type)

    def __repr__(self):
        return self.anchor_type.name


class RegexParser:
    def __init__(self, regex: str):
        self._regex = regex
        self._pos = 0
        self._root = self.parse_regex()
        if self._pos < len(self._regex):
            raise ValueError(
                f"could not finish parsing regex, left = {self._regex[self._pos:]}"
            )

    @property
    def root(self):
        return self._root

    def consume(self, char):
        if self._pos >= len(self._regex):
            raise ValueError("index out of bounds")
        if self.current() != char:
            raise ValueError(f"expected {char} got {self.current()}")
        self._pos += 1

    def consume_and_return(self):
        char = self.current()
        self.consume(char)
        return char

    def optional(self, expected: str) -> bool:
        if self.matches(expected):
            self.consume(expected)
            return True
        return False

    def current(self, lookahead=None):
        if lookahead is not None:
            return self._regex[self._pos + lookahead]
        return self._regex[self._pos]

    def remainder(self):
        return "" if self._pos >= len(self._regex) else self._regex[self._pos :]

    def parse_regex(self) -> RegexNode:
        if self._regex == "":
            raise ValueError(f"regex is empty")

        if self.matches("^"):
            anchor = Anchor(self._pos, AnchorType.get(self.consume_and_return()))
            if self.remainder():
                expr = self.parse_expression()
                expr.seq.insert(0, anchor)
                return expr
            else:
                return anchor
        return self.parse_expression()

    def can_parse_group(self):
        return self.current() == "("

    def can_parse_char(self):
        return self._pos < len(self._regex) and self.current() not in ESCAPED

    def can_parse_match(self):
        return (
            self.matches(".")
            or self.can_parse_character_class_or_group()
            or self.can_parse_char()
            or self.can_parse_escaped()
        )

    def inbound(self, lookahead=0):
        return self._pos + lookahead < len(self._regex)

    def can_parse_sub_expression_item(self):
        return self.inbound() and (
            self.can_parse_group() or self.can_parse_anchor() or self.can_parse_match()
        )

    def matches(self, char):
        return self.inbound() and self.current() == char

    def matches_any(self, options, lookahead: int = 0):
        return self.inbound(lookahead) and self.current(lookahead) in options

    def parse_expression(self) -> Expression:
        # Expression ::= Subexpression ("|" Expression)?
        pos = self._pos
        sub_exprs = self.parse_sub_expression()
        expr = None
        if self.matches("|"):
            self.consume("|")
            expr = (
                self.parse_expression()
                if self.can_parse_sub_expression_item()
                else Anchor.empty_string(self._pos)
            )
        return Expression(pos, sub_exprs, expr)

    def parse_sub_expression(self) -> list[SubExpressionItem]:
        # Subexpression ::= SubexpressionItem+
        sub_exprs = [self.parse_sub_expression_item()]
        while self.can_parse_sub_expression_item():
            sub_exprs.append(self.parse_sub_expression_item())
        return sub_exprs

    def parse_sub_expression_item(self) -> SubExpressionItem:
        if self.matches("("):
            return self.parse_group()
        elif self.can_parse_anchor():
            return self.parse_anchor()
        else:
            return self.parse_match()

    def parse_group(self) -> Group | Expression:
        self.consume("(")
        is_capturing = True
        if self._regex.startswith("?:"):
            self.consume("?:")
            is_capturing = False
        expr = self.parse_expression()
        self.consume(")")
        quantifier = None
        if self.can_parse_quantifier():
            quantifier = self.parse_quantifier()
            # handle range qualifies and return a list of matches instead
            if isinstance(quantifier.item, RangeQuantifier):
                return quantifier.item.expand(expr, quantifier.lazy)
        return Group(self._pos, expr, quantifier, is_capturing)

    def can_parse_quantifier(self):
        return self.matches_any(("*", "+", "?", "{"))

    def parse_quantifier(self):
        if self.matches_any(("*", "+", "?")):
            quantifier_item = QuantifierChar(
                QuantifierType.get(self.consume_and_return())
            )
        else:
            quantifier_item = self.parse_range_quantifier()
        return Quantifier(quantifier_item, self.optional("?"))

    def parse_int(self):
        digits = []
        while self.current().isdigit():
            digits.append(self.consume_and_return())
        return int("".join(digits))

    def parse_range_quantifier(self) -> RangeQuantifier:
        # RangeQuantifier ::= "{" RangeQuantifierLowerBound ( "," RangeQuantifierUpperBound? )? "}"
        self.consume("{")
        # RangeQuantifierLowerBound = Integer
        lower_bound = 0 if self.matches(",") else self.parse_int()
        upper_bound = None
        while self.current() == ",":
            upper_bound = inf
            self.consume_and_return()
            if self.current().isdigit():
                upper_bound = self.parse_int()
        self.consume("}")
        return RangeQuantifier(lower_bound, upper_bound)

    def parse_match(self):
        # Match ::= MatchItem Quantifier?
        pos = self._pos
        match_item = self.parse_match_item()
        quantifier = None
        if self.can_parse_quantifier():
            quantifier = self.parse_quantifier()
            # handle range qualifies and return a list of matches instead
            if isinstance(quantifier.item, RangeQuantifier):
                return quantifier.item.expand(match_item, quantifier.lazy)
        return Match(pos, match_item, quantifier)

    def can_parse_character_group(self):
        return self.matches("[")

    def parse_character_class(self) -> CharacterGroup:
        self.consume("\\")
        if self.matches_any(("w", "W")):
            return CharacterGroup(
                self._pos,
                (
                    CharacterRange("A", "Z"),
                    CharacterRange("a", "z"),
                    CharacterScalar(self._pos, "_"),
                ),
                self.matches("W"),
            )
        elif self.matches_any(("d", "D")):
            return CharacterGroup(
                self._pos, (CharacterRange("0", "9"), self.matches("D"))
            )
        elif self.matches_any(("s", "S")):
            return CharacterGroup(
                self._pos,
                tuple(
                    map(
                        lambda c: CharacterScalar(self._pos, c),
                        [" ", "\t", "\n", "\r", "\v", "\f"],
                    )
                ),
                self.matches("S"),
            )
        else:
            raise ValueError(f"unrecognized character class{self.current()}")

    def parse_character_range(self, char: str) -> CharacterRange:
        self.consume("-")
        to = self.parse_char()
        assert to.char != "]"
        return CharacterRange(char, to.char)

    def can_parse_character_class(self):
        return self.matches("\\") and self.matches_any(character_classes, 1)

    def parse_character_group_item(self) -> CharacterGroupItem | CharacterGroup:
        if self.can_parse_character_class():
            return self.parse_character_class()
        else:
            # If the dash character is the first one in the list,
            # then it is treated as an ordinary character.
            # For example [-AZ] matches '-' or 'A' or 'Z' .
            # And tag[-]line matches "tag-line" and "tag line" as in a previous example.

            if self.matches_any(no_escape_in_group):
                if self.matches("\\"):
                    self.consume("\\")
                return CharacterScalar(self._pos, self.consume_and_return())
            char = self.parse_char()
            if self.matches("-"):
                return self.parse_character_range(char.char)
            else:
                return char

    def parse_character_group(self):
        # CharacterGroup ::= "[" CharacterGroupNegativeModifier? CharacterGroupItem+ "]"
        self.consume("[")
        negated = False
        if self.matches("^"):
            self.consume("^")
            negated = True
        items = []
        group_pos = self._pos
        while self.can_parse_char() or self.matches_any(no_escape_in_group):
            items.append(self.parse_character_group_item())
        if not items:
            raise ValueError(
                f"failed parsing from {group_pos}: {self._regex[group_pos:]}"
            )
        self.consume("]")
        return CharacterGroup(group_pos, tuple(items), negated)

    def parse_char(self):
        if self.can_parse_escaped():
            return self.parse_escaped()
        if not self.can_parse_char():
            raise ValueError(
                f"expected a char: found "
                f'{"EOF" if self._pos >= len(self._regex) else self.current()} at index {self._pos}'
            )
        return CharacterScalar(self._pos - 1, self.consume_and_return())

    def can_parse_escaped(self):
        return self.matches("\\") and self.matches_any(ESCAPED, 1)

    def can_parse_anchor(self):
        return (
            self.matches("\\") and self.matches_any({"A", "z", "Z", "G", "b", "B"}, 1)
        ) or self.matches_any(("^", "$"))

    def parse_escaped(self):
        self.consume("\\")
        return CharacterScalar(self._pos - 1, self.consume_and_return())

    def can_parse_character_class_or_group(self):
        return self.can_parse_character_class() or self.can_parse_character_group()

    def parse_character_class_or_group(self):
        if self.can_parse_character_class():
            return self.parse_character_class()
        else:
            return self.parse_character_group()

    def parse_match_item(self):
        if self.matches("."):  # parse AnyCharacter
            self.consume(".")
            return MatchAnyCharacter(self._pos)
        elif self.can_parse_character_class_or_group():
            return self.parse_character_class_or_group()
        else:
            return self.parse_char()

    def parse_anchor(self):
        pos = self._pos
        if self.matches("\\"):
            self.consume("\\")
            assert self.current() in {"A", "z", "Z", "G", "b", "B"}
        return Anchor(pos, AnchorType.get(self.consume_and_return()))

    def __repr__(self):
        return f"Parser({self._regex})"
