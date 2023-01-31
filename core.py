from abc import ABC, abstractmethod
from collections import defaultdict
from copy import copy
from dataclasses import astuple, dataclass, field
from enum import Enum, IntFlag, auto
from itertools import chain, combinations, count, product
from string import ascii_uppercase
from sys import maxsize
from typing import Hashable, Iterable, Iterator, Optional, Union

import graphviz
from more_itertools import first, first_true, minmax, pairwise
from ordered_set import OrderedSet

from unionfind import UnionFind

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
        if any(s == src_fsm.start for s in sources):
            dst_fsm.start = strid
        if any(s in src_fsm.accept for s in sources):
            dst_fsm.accept.add(strid)
    sourcescache[sources] = strid
    return strid


INLINE_MODIFIER_START = "(?"


class InvalidCharacterRange(Exception):
    ...


class RegexFlag(IntFlag):
    NOFLAG = auto()
    IGNORECASE = auto()
    MULTILINE = auto()
    DOTALL = auto()  # make dot match newline
    FREESPACING = auto()


class TagType(Enum):
    Epsilon = "ε"
    GroupEntry = "GroupEntry"
    GroupExit = "GroupExit"
    GroupLink = ""
    Fence = "Fence"


class Matchable(Hashable):
    @abstractmethod
    def match(self, text: str, position: int, flags: RegexFlag) -> bool:
        ...

    def is_opening_group(self):
        return isinstance(self, Tag) and self.tag_type == TagType.GroupEntry

    def is_closing_group(self):
        return isinstance(self, Tag) and self.tag_type == TagType.GroupExit


class Virtual(Matchable, ABC):
    ...


@dataclass
class Tag(Virtual):
    tag_type: TagType
    group_index: int
    substr: str

    @staticmethod
    def entry(group_index: int, substr: str) -> "Tag":
        return Tag(TagType.GroupEntry, group_index, substr)

    @staticmethod
    def exit(group_index: int, substr: str) -> "Tag":
        return Tag(TagType.GroupExit, group_index, substr)

    @staticmethod
    def link() -> "Tag":
        return Tag(TagType.GroupLink, maxsize, "")

    @staticmethod
    def barrier() -> "Tag":
        return Tag(TagType.Fence, maxsize, "")

    @staticmethod
    def epsilon() -> "Tag":
        return Tag(TagType.Epsilon, maxsize, "")

    def match(self, text: str, position: int, flags: RegexFlag) -> bool:
        if self.tag_type == TagType.GroupLink:
            return False
        return True

    def __hash__(self):
        return hash((self.tag_type, self.group_index, self.substr))

    def __repr__(self):
        match self.tag_type:
            case TagType.Fence | TagType.Epsilon:
                return self.tag_type.value
            case TagType.GroupLink:
                return ""
            case TagType.GroupEntry | TagType.GroupExit:
                return f"{self.tag_type.name}({self.group_index})"
            case _:
                raise NotImplementedError


@dataclass(frozen=True)
class Fragment:
    start: State = field(default_factory=gen_state)
    end: State = field(default_factory=gen_state)

    def __iter__(self):
        yield from astuple(self)


@dataclass()
class Transition:
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


class NFA(defaultdict[State, list[Transition]]):
    """Formally, an NFA is a 5-tuple (Q, Σ, q0, T, δ) where
        • Q is finite set of states;
        • Σ is alphabet of input symbols;
        • q0 is start state;
        • T is subset of Q giving the ``accept`` states;
        and
        • δ is the transition function.
    Now the transition function specifies a set of states rather than a state: it maps Q × Σ to { subsets of Q }."""

    def __init__(self):
        super(NFA, self).__init__(list)
        self.symbols: set[Matchable] = set()
        self.states: set[State] = set()
        self.accept: set[State] = set()
        self.start: State = -1

    def update_symbols_and_states(self):
        for start, transitions in self.items():
            self.states.add(start)
            for symbol, end in transitions:
                self.states.add(end)
                self.symbols.add(symbol)
        self.symbols.discard(Tag.epsilon())

    def update_symbols(self):
        for transition in chain.from_iterable(self.values()):
            self.symbols.add(transition.matchable)
        self.symbols.discard(Tag.epsilon())

    def _dict(self) -> defaultdict[State, list[Transition]]:
        t = defaultdict(list)
        for state, transitions in self.items():
            t[state] = transitions.copy()
        return t

    def set_start(self, state: State):
        self.start = state

    def set_terminals(self, fragment: Fragment):
        self.set_start(fragment.start)
        self.set_accept(fragment.end)

    def set_accept(self, accept: State):
        self.accept = {accept}

    def transition(
        self, state: State, symbol: Matchable, _: bool = False
    ) -> tuple[State, ...]:
        result = tuple(
            transition.end
            for transition in self[state]
            if transition.matchable == symbol
        )
        return result

    def add_transition(self, start: State, end: State, matchable: Matchable):
        self[start].append(Transition(matchable, end))

    def reverse_transitions(self, state: State):
        self[state].reverse()

    def epsilon(self, start: State, end: State):
        self.add_transition(start, end, Tag.epsilon())

    def __repr__(self):
        return (
            f"FSM(states={self.states}, "
            f"symbols={self.symbols}, "
            f"start_state={self.start}, "
            f"accept_states={self.accept}) "
        )

    def symbol_closure(
        self, states: Iterable[State], collapsed: Tag = Tag.epsilon()
    ) -> tuple[State, ...]:
        """
        This is the set of all the nodes which can be reached by following epsilon labeled edges
        This is done here using a depth first search

        https://castle.eiu.edu/~mathcs/mat4885/index/Webview/examples/epsilon-closure.pdf
        """

        seen = set()
        stack = list(states)
        closure = OrderedSet()

        while stack:
            if (state := stack.pop()) in seen:
                continue

            seen.add(state)
            # explore the states in the order which they are in
            nxt = self.transition(state, collapsed, True)[::-1]
            stack.extend(nxt)
            closure.add(state)

        return tuple(closure)

    def move(self, states: Iterable[State], symbol: Matchable) -> frozenset[State]:
        s = set()
        for state in states:
            item = self.transition(state, symbol)
            if item and isinstance(item, str):
                s.update({item})
            else:
                s.update(item)
        return frozenset(s)

    def zero_or_more(self, fragment: Fragment, lazy: bool) -> Fragment:
        empty_fragment = self.base(Anchor.empty_string())

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
        self.add_transition(fragment.start, fragment.end, Anchor.empty_string())
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
        dot.attr("node", fontname="Palatino")
        dot.attr("edge", fontname="Palatino")

        seen = set()

        for state, transitions in self.items():
            for transition in transitions:
                symbol, end = transition
                if state not in seen:
                    color = "green" if state == self.start else "gray"
                    dot.node(
                        str(state),
                        color=color,
                        shape="doublecircle" if state in self.accept else "circle",
                        style="filled",
                    )
                    seen.add(state)
                if end not in seen:
                    color = "gray"
                    dot.node(
                        f"{end}",
                        color=color,
                        shape="doublecircle" if end in self.accept else "circle",
                        style="filled",
                    )
                    seen.add(end)
                if isinstance(symbol, Tag) and symbol.tag_type == TagType.GroupLink:
                    dot.edge(str(state), str(end), color="blue", style="dotted")
                else:
                    dot.edge(str(state), str(end), label=str(symbol), color="black")

        dot.node("start", shape="none")
        dot.edge("start", f"{self.start}", arrowhead="vee")
        dot.render(view=True, directory="graphs", filename=str(id(self)))


class DFA(NFA):
    def __init__(self, nfa: Optional[NFA] = None):
        super(DFA, self).__init__()
        if nfa is not None:
            self.subset_construction(nfa, Tag.epsilon())

    def transition(
        self, state: State, symbol: Matchable, wrapped: bool = False
    ) -> State:
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
        cp.start = self.start
        cp.accept = self.accept.copy()
        cp.states = self.states.copy()
        return cp

    def subset_construction(self, nfa: NFA, collapsed):
        seen, stack = set(), [(nfa.start,)]

        finished = []

        while stack:
            nfa_states = stack.pop()  # what is the epsilon closure of the dfa_states
            closure = nfa.symbol_closure(nfa_states, collapsed)
            start = gen_dfa_state(closure, src_fsm=nfa, dst_fsm=self)
            # next we want to see which states are reachable from each of the states in the epsilon closure
            for symbol in nfa.symbols - {collapsed}:
                if move := nfa.symbol_closure(nfa.move(closure, symbol), collapsed):
                    end = gen_dfa_state(move, src_fsm=nfa, dst_fsm=self)
                    self.add_transition(start, end, symbol)
                    if move not in seen:
                        seen.add(move)
                        stack.append(move)
            finished.append(start)

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
            if (p in self.accept) == (q in self.accept):
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
        accept_states = self.accept.copy()
        states = set()
        lost = {}

        for sources in self.gen_equivalence_states():
            compound = gen_dfa_state(sources, src_fsm=self, dst_fsm=self)
            states.add(compound)
            for original in sources:
                lost[original] = compound

        self.states = states
        self.accept = self.accept - accept_states

        for start in tuple(self):
            if start in lost:
                self[lost.get(start)] = self.pop(start)

        for start in self:
            new_transitions = set()
            for transition in tuple(self[start]):
                if transition.end in lost:
                    new_transitions.add(
                        Transition(transition.matchable, lost.get(transition.end))
                    )
                else:
                    new_transitions.add(transition)
            self[start] = new_transitions

        cp = self.copy()
        self.clear()
        self.subset_construction(cp, Tag.barrier())

    def clear(self) -> None:
        super().clear()
        self.symbols.clear()
        self.start = -1
        self.accept.clear()
        self.states.clear()

    def add_transition(self, start: State, end: State, matchable: Matchable) -> None:
        self[start].append(Transition(matchable, end))


ESCAPED = set(". \\ + * ? [ ^ ] $ ( ) { } = < > | -".split())

CHARACTER_CLASSES = {"w", "W", "s", "S", "d", "D"}

UNESCAPED_IN_CHAR_GROUP = ESCAPED - {"]"}


@dataclass
class RegexNode(ABC):
    pos: int = field(repr=False)

    @abstractmethod
    def fsm(self, nfa: NFA) -> Fragment:
        ...

    @abstractmethod
    def string(self) -> str:
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

    def string(self):
        lazy = "?" if self.lazy else ""
        match self.item.type:
            case QuantifierType.OneOrMore:
                return "+" + lazy
            case QuantifierType.ZeroOrMore:
                return "*" + lazy
            case QuantifierType.ZeroOrOne:
                return "?" + lazy
            case _:
                raise NotImplementedError


@dataclass
class RangeQuantifier(QuantifierItem):
    start: int
    end: Optional[int] = None

    def __post_init__(self):
        if self.end is None:
            if self.start < 0:
                raise InvalidCharacterRange(
                    f"fixed quantifier, {{n}} must be >= 0: not {self.start}"
                )
        elif isinstance(self.end, int):
            if self.end == maxsize:
                if self.start < 0:
                    raise InvalidCharacterRange(
                        f"for {{n,}} quantifier, {{n}} must be >= 0: not {self.start}"
                    )
            else:
                if self.start < 0:
                    raise InvalidCharacterRange(
                        f"for {{n, m}} quantifier, {{n}} must be >= 0: not {self.start}"
                    )
                if self.end < self.start:
                    raise InvalidCharacterRange(
                        f"for {{n, m}} quantifier, {{m}} must be >= {{n}}: not {self.end}"
                    )
        elif self.start == 0:
            if not isinstance(self.end, int):
                raise InvalidCharacterRange(f"invalid upper bound {self.end}")
            if self.end < 1:
                raise InvalidCharacterRange(
                    f"for {{, m}} quantifier, {{m}} must be >= 1: not {self.end}"
                )
        else:
            raise InvalidCharacterRange(f"invalid range {{{self.start}, {self.end}}}")

    def expand(
        self,
        item: Union["SubExpressionItem", "Expression"],
        lazy: bool,
        group_index: Optional[int] = None,
    ):
        # e{3} expands to eee; e{3,5} expands to eeee?e?, and e{3,} expands to eee+.
        # e{0} expands to ''
        if self.start == 0 and self.end is None:
            return Anchor.empty_string()

        seq = [copy(item) for _ in range(self.start)]

        if self.end is not None:
            if self.end == maxsize:
                if self.start > 0:
                    item = seq.pop()
                    seq.append(
                        Group(
                            item.pos,
                            item,
                            Quantifier(QuantifierChar(QuantifierType.OneOrMore), lazy),
                            group_index=group_index,
                            substr=None,
                        )
                    )
                else:
                    seq.append(
                        Group(
                            item.pos,
                            item,
                            Quantifier(QuantifierChar(QuantifierType.ZeroOrMore), lazy),
                            group_index=group_index,
                            substr=None,
                        )
                    )

            else:
                for _ in range(self.start, self.end):
                    seq.append(
                        Group(
                            item.pos,
                            item,
                            Quantifier(QuantifierChar(QuantifierType.ZeroOrOne), lazy),
                            group_index=group_index,
                            substr=None,
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

    def string(self):
        seq = "".join(item.string() for item in self.seq)
        if self.alternate is not None:
            return f"{seq}|{self.alternate.string()}"


@dataclass
class Group(SubExpressionItem):
    expression: Expression
    quantifier: Optional[Quantifier]
    group_index: Optional[int]
    substr: Optional[str]

    def capturing(self):
        return self.group_index is not None

    def fsm(self, nfa: NFA) -> Fragment:
        fragment = self.expression.fsm(nfa)
        if self.group_index is not None:
            state1, state2, state3 = gen_state(), gen_state(), gen_state()
            nfa.base(
                Tag.entry(self.group_index, self.substr),
                Fragment(state1, fragment.start),
            )
            nfa.base(
                Tag.exit(self.group_index, self.substr),
                Fragment(fragment.end, state2),
            )
            nfa.add_transition(state2, state3, Tag.barrier())
            nfa.add_transition(state2, fragment.start, Tag.link())
            fragment = Fragment(state1, state3)
        if self.quantifier:
            fragment = self.quantifier.apply(fragment, nfa)
        return fragment

    def string(self):
        expression = f"({self.expression.string()})"
        if self.quantifier is not None:
            return f"{expression}{self.quantifier.string()}"
        return expression


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

    def string(self):
        return f'{self.item.string()}{self.quantifier if self.quantifier else ""}'


@dataclass
class MatchAnyCharacter(MatchItem, Matchable):
    ignore: tuple = ("\n",)

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

    def string(self):
        return "."


class CharacterGroupItem(Matchable, ABC):
    pass


@dataclass
class CharacterScalar(CharacterGroupItem, MatchItem):
    char: str

    def fsm(self, nfa: NFA) -> Fragment:
        fragment = Fragment()
        nfa.add_transition(fragment.start, fragment.end, self)
        return fragment

    def match(self, text, position, flags) -> bool:
        if position < len(text):
            if flags & RegexFlag.IGNORECASE:
                return self.char.casefold() == text[position].casefold()
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

    def string(self):
        return self.char


class MatchCharacterClass(MatchItem, ABC):
    pass


@dataclass
class CharacterGroup(MatchCharacterClass, Matchable):
    items: tuple[CharacterGroupItem, ...]
    negated: bool = False

    def fsm(self, nfa: NFA) -> Fragment:
        fragment = Fragment()
        nfa.add_transition(fragment.start, fragment.end, self)
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
        return f"[{('^' if self.negated else '')}{''.join(map(repr, self.items))}]"

    def __lt__(self, other):
        return id(self) < id(other)

    def __hash__(self):
        return hash((self.items, self.negated))

    def string(self):
        return self.__repr__()


@dataclass
class CharacterRange(CharacterGroupItem, Matchable):
    start: str
    end: str

    def match(self, text, position, flags) -> bool:
        if position < len(text):
            if flags & RegexFlag.IGNORECASE:
                return (
                    self.start.casefold()
                    <= text[position].casefold()
                    <= self.end.casefold()
                )
            else:
                return self.start <= text[position] <= self.end
        return False

    def __post_init__(self):
        if self.start > self.end:
            raise InvalidCharacterRange(f"[{self.start}-{self.end}] is not ordered")

    def __hash__(self):
        return hash((self.start, self.end))

    def __repr__(self):
        return f"[{self.start}-{self.end}]"


class AnchorType(Enum):
    Start = "^"
    End = "$"
    Empty = ""

    # must be escaped
    WordBoundary = "\\b"
    NonWordBoundary = "\\B"
    AnchorStartOfStringOnly = "\\A"
    AnchorEndOfStringOnlyNotNewline = "\\z"
    AnchorEndOfStringOnly = "\\Z"
    AnchorPreviousMatchEnd = "\\G"

    @staticmethod
    def get(char):
        match char:
            case "^":
                return AnchorType.Start
            case "$":
                return AnchorType.End
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
class Anchor(SubExpressionItem, Virtual):
    anchor_type: AnchorType

    def fsm(self, nfa: NFA) -> Fragment:
        return nfa.base(self)

    @staticmethod
    def empty_string(pos: int = maxsize) -> "Anchor":
        return Anchor(pos, AnchorType.Empty)

    def match(self, text, position, flags) -> bool:
        match self.anchor_type:
            case AnchorType.Start:
                # assert that this is the beginning of the string
                return position == 0
                # or the previous char is a \n if MULTILINE mode enabled
            case AnchorType.End:
                return (
                    position >= len(text)
                    or position == len(text) - 1
                    and text[position] == "\n"
                )
            case AnchorType.WordBoundary:
                return is_word_boundary(text, position)
            case AnchorType.NonWordBoundary:
                return text and not is_word_boundary(text, position)
            case AnchorType.Empty:
                return True

        raise NotImplementedError

    def string(self):
        return self.anchor_type.value

    def __hash__(self):
        return hash(self.anchor_type)

    def __repr__(self):
        return self.anchor_type.name


class RegexpParser:
    def __init__(self, regex: str):
        self._regex = regex
        self._pos = 0
        self._flags = RegexFlag.NOFLAG
        self._group_count = 0
        self._root = self.parse_regex()
        if self._pos < len(self._regex):
            raise ValueError(
                f"could not finish parsing regex, left = {self._regex[self._pos:]}"
            )

    @property
    def group_count(self):
        return self._group_count

    @property
    def root(self):
        return self._root

    def consume(self, char: str):
        if self._pos >= len(self._regex):
            raise ValueError("index out of bounds")
        if not self.remainder().startswith(char):
            raise ValueError(
                f"expected {char} got {self.current()}\n"
                f"regexp = {self._regex!r}\n"
                f"left = {(' ' * (self._pos + 4) + self.remainder())!r}"
            )
        self._pos += len(char)

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

    def parse_inline_modifiers(self):
        modifiers = []
        allowed = ("i", "m", "s", "x")

        while self.remainder().startswith(INLINE_MODIFIER_START):
            if not self.matches_any(allowed, len(INLINE_MODIFIER_START)):
                break
            self.consume(INLINE_MODIFIER_START)
            while self.matches_any(allowed):
                modifiers.append(self.consume_and_return())
            self.consume(")")

        for modifier in modifiers:
            match modifier:
                case "i":
                    self._flags |= RegexFlag.IGNORECASE
                case "s":
                    self._flags |= RegexFlag.DOTALL
                case "m":
                    self._flags |= RegexFlag.MULTILINE
                case "x":
                    self._flags |= RegexFlag.FREESPACING
                case "_":
                    raise ValueError()

    def parse_regex(self) -> RegexNode:
        if self._regex == "":
            return Anchor.empty_string()
        self.parse_inline_modifiers()

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
        return self.matches("(")

    def can_parse_char(self):
        return self._pos < len(self._regex) and self.current() not in ESCAPED

    def can_parse_match(self):
        return (
            self.matches(".")
            or self.can_parse_character_class_or_group()
            or self.can_parse_char()
            or self.can_parse_escaped()
        )

    def within_bounds(self, lookahead: int = 0) -> bool:
        return self._pos + lookahead < len(self._regex)

    def can_parse_sub_expression_item(self):
        return self.within_bounds() and (
            self.can_parse_group() or self.can_parse_anchor() or self.can_parse_match()
        )

    def matches(self, char):
        return self.within_bounds() and self.current() == char

    def matches_any(self, options, lookahead: int = 0):
        return self.within_bounds(lookahead) and self.current(lookahead) in options

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
        start = self._pos
        self.consume("(")
        index = self._group_count
        self._group_count += 1
        if self.remainder().startswith("?:"):
            self.consume("?:")
            index = None
            self._group_count -= 1
        if self.matches(")"):
            expr = Anchor.empty_string()
        else:
            expr = self.parse_expression()
        self.consume(")")
        end = self._pos
        quantifier = None
        if self.can_parse_quantifier():
            quantifier = self.parse_quantifier()
            # handle range qualifies and return a list of matches instead
            if isinstance(quantifier.item, RangeQuantifier):
                return quantifier.item.expand(expr, quantifier.lazy, index)
        return Group(self._pos, expr, quantifier, index, self._regex[start:end])

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
        lower = 0 if self.matches(",") else self.parse_int()
        upper = None
        while self.current() == ",":
            upper = maxsize
            self.consume_and_return()
            if self.current().isdigit():
                upper = self.parse_int()
        self.consume("}")
        return RangeQuantifier(lower, upper)

    def parse_match(self) -> Match | Expression:
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
        return self.matches("\\") and self.matches_any(CHARACTER_CLASSES, 1)

    def parse_character_group_item(self) -> CharacterGroupItem | CharacterGroup:
        if self.can_parse_character_class():
            return self.parse_character_class()
        else:
            # If the dash character is the first one in the list,
            # then it is treated as an ordinary character.
            # For example [-AZ] matches '-' or 'A' or 'Z' .
            # And tag[-]line matches "tag-line" and "tag line" as in a previous example.

            if self.matches_any(UNESCAPED_IN_CHAR_GROUP):
                if self.matches("\\"):
                    self.consume("\\")
                return CharacterScalar(self._pos, self.consume_and_return())
            char = self.parse_char()
            if self.matches("-"):
                return self.parse_character_range(char.char)
            else:
                return char

    def save_state(self) -> tuple[int, RegexFlag]:
        return self._pos, self.flags

    def parse_character_group(self):
        # CharacterGroup ::= "[" CharacterGroupNegativeModifier? CharacterGroupItem+ "]"
        self.consume("[")
        negated = False
        if self.matches("^"):
            self.consume("^")
            negated = True
        state = self.save_state()
        items = []
        try:
            while self.can_parse_char() or self.matches("\\"):
                items.append(self.parse_character_group_item())
            self.consume("]")
        except ValueError:
            self._pos, self._flags = state
            while self.can_parse_char() or self.matches_any(UNESCAPED_IN_CHAR_GROUP):
                items.append(CharacterScalar(self._pos, self.consume_and_return()))
            self.consume("]")

        if not items:
            raise ValueError(
                f"failed parsing from {state[0]}\n"
                f"regexp = {self._regex}\n"
                f"left   = {' ' * self._pos + self._regex[self._pos:]}"
            )

        return CharacterGroup(state[0], tuple(items), negated)

    def parse_char(self):
        if self.can_parse_escaped():
            return self.parse_escaped()
        if not self.can_parse_char():
            raise ValueError(
                f"expected a char: found {self.current() if self.within_bounds() else 'EOF'}\n"
                f"regexp = {self._regex}\n"
                f"left   = {' ' * self._pos + self.remainder()}"
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

    @property
    def flags(self):
        return self._flags
