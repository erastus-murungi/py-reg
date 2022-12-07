import sys
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass, field
from enum import Enum
from math import inf
from typing import MutableMapping, Optional, Sequence

from core import (CompoundMatchableMixin, MatchableMixin, State, T,
                  TransitionsProvider)

ESCAPED = set(". \\ + * ? [ ^ ] $ ( ) { } = ! < > | : -".split())

anchors = {"$", "^"}

character_classes = {"w", "W", "s", "S", "d", "D"}

StatePair = tuple[State, State]


def zero_or_more(state_pair, transitions) -> StatePair:
    source, sink = state_pair
    new_start, new_accept = State.pair()
    transitions[sink][Epsilon].add(source)
    transitions[new_start][Epsilon].add(source)
    transitions[sink][Epsilon].add(new_accept)
    transitions[new_start][Epsilon].add(new_accept)
    return new_start, new_accept


def one_or_more(state_pair: StatePair, transitions):
    source, sink = state_pair
    transitions[sink][Epsilon].add(source)
    return source, sink


def zero_or_one(state_pair: StatePair, transitions):
    source, sink = state_pair
    transitions[source][Epsilon].add(sink)
    return source, sink


def alternation(lower, upper, transitions):
    lower_start, lower_accept = lower
    upper_start, upper_accept = upper
    new_start, new_accept = State.pair()

    transitions[new_start][Epsilon].add(lower_start)
    transitions[new_start][Epsilon].add(upper_start)
    transitions[lower_accept][Epsilon].add(new_accept)
    transitions[upper_accept][Epsilon].add(new_accept)

    return new_start, new_accept


def concatenate(state_pair1_end, state_pair2_start, transitions):
    transitions[state_pair1_end][Epsilon].add(state_pair2_start)


def trivial(
    matchable: MatchableMixin,
    transitions: MutableMapping[State, TransitionsProvider],
) -> StatePair:
    source, sink = State.pair()
    transitions[source][matchable].add(sink)
    return source, sink


@dataclass
class RegexNode(ABC):
    pos: int = field(repr=False)

    @abstractmethod
    def to_fsm(
        self, transitions: MutableMapping[State, TransitionsProvider]
    ) -> StatePair:
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

    def transform(self, state_pair, transitions):
        if isinstance(self.item, QuantifierChar):
            match self.item.type:
                case QuantifierType.OneOrMore:
                    return one_or_more(state_pair, transitions)
                case QuantifierType.ZeroOrMore:
                    return zero_or_more(state_pair, transitions)
                case QuantifierType.ZeroOrOne:
                    return zero_or_one(state_pair, transitions)
        raise NotImplementedError


class HasQuantifier(ABC):
    @abstractmethod
    def get_quantifier(self) -> Optional[Quantifier]:
        pass

    @abstractmethod
    def set_quantifier(self, quantifier: Quantifier):
        pass


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

    def expand(self, item: "SubExpressionItem"):
        # e{3} expands to eee; e{3,5} expands to eeee?e?, and e{3,} expands to eee+.
        assert isinstance(item, (RegexNode, HasQuantifier))

        seq = []
        for _ in range(self.start):
            seq.append(copy(item))

        if self.end is not None:
            if self.end == inf:
                item = seq.pop()
                seq.append(
                    Group(
                        item.pos,
                        Expression(item.pos, [item]),
                        Quantifier(QuantifierChar(QuantifierType.OneOrMore)),
                    )
                )
            else:
                for _ in range(self.start, self.end):
                    seq.append(
                        Group(
                            item.pos,
                            Expression(item.pos, [item]),
                            Quantifier(QuantifierChar(QuantifierType.ZeroOrOne)),
                        )
                    )
        return Expression(item.pos, seq)


class SubExpressionItem(RegexNode, ABC):
    pass


@dataclass
class Expression(RegexNode):
    seq: list[SubExpressionItem]
    alternate: Optional["Expression"] = None

    def to_fsm(self, transitions) -> StatePair:
        nodes = [subexpression.to_fsm(transitions) for subexpression in self.seq]
        for (_, s1_accept), (s2_start, _) in zip(nodes, nodes[1:]):
            transitions[s1_accept][Epsilon].add(s2_start)
        state_pair = nodes[0][0], nodes[-1][1]
        if self.alternate is None:
            return state_pair
        return alternation(state_pair, self.alternate.to_fsm(transitions), transitions)


@dataclass
class Group(SubExpressionItem, HasQuantifier):
    expression: Expression
    quantifier: Optional[Quantifier]
    is_capturing: bool = False

    def to_fsm(
        self, transitions: MutableMapping[State, TransitionsProvider]
    ) -> StatePair:
        state_pair = self.expression.to_fsm(transitions)
        if self.quantifier:
            return self.quantifier.transform(state_pair, transitions)
        return state_pair

    def set_quantifier(self, quantifier: Quantifier):
        self.quantifier = quantifier

    def get_quantifier(self) -> Optional[Quantifier]:
        return self.quantifier


class MatchItem(SubExpressionItem, ABC):
    pass


@dataclass
class Match(SubExpressionItem, HasQuantifier):
    item: MatchItem
    quantifier: Optional[Quantifier]

    def to_fsm(self, transitions: dict[State, TransitionsProvider]) -> StatePair:
        state_pair = self.item.to_fsm(transitions)
        if self.quantifier is None:
            return state_pair
        return self.quantifier.transform(state_pair, transitions)

    def get_quantifier(self) -> Optional[Quantifier]:
        return self.quantifier

    def set_quantifier(self, quantifier: Quantifier):
        self.quantifier = quantifier


@dataclass
class MatchAnyCharacter(MatchItem, CompoundMatchableMixin):
    ignore: tuple = ("ε", "\n")

    def to_fsm(self, transitions: dict[State, TransitionsProvider]) -> StatePair:
        return trivial(self, transitions)

    def __eq__(self, other):
        return isinstance(other, MatchAnyCharacter) and other.ignore == self.ignore

    def match(self, position: int, text: Sequence[T]) -> bool:
        return position < len(text) and text[position] not in self.ignore

    def __repr__(self):
        return "Any"

    def __hash__(self):
        return hash(".") ^ 12934


@dataclass
class Char(MatchableMixin, MatchItem):
    char: str

    def to_fsm(self, transitions) -> tuple[State, State]:
        source, sink = State.pair()
        transitions[source][self].add(sink)
        return source, sink

    def match(self, position: int, text: Sequence[T]) -> bool:
        return self.char == text[position]

    def __eq__(self, other) -> bool:
        return other == self.char

    def __lt__(self, other) -> bool:
        if isinstance(other, Char):
            return self.char <= other.char
        return other <= self.char

    def __repr__(self):
        return f"{self.char}"

    def __hash__(self):
        return hash(self.char)


Epsilon = Char(-sys.maxsize, "ε")


class MatchCharacterClass(MatchItem, ABC):
    pass


class CharacterGroupItem(ABC):
    pass


@dataclass
class CharacterGroup(MatchCharacterClass, CompoundMatchableMixin):
    items: list[CharacterGroupItem]
    negated: bool = False

    intervals: frozenset[tuple[str, str]] = field(default_factory=frozenset, repr=False)
    options: frozenset[str] = field(default_factory=frozenset, repr=False)

    def to_fsm(
        self, transitions: dict[State, TransitionsProvider]
    ) -> tuple[State, State]:
        source, sink = State.pair()
        transitions[source][self].add(sink)
        return source, sink

    def __post_init__(self):
        self.intervals = frozenset(
            [
                (item.start.char, item.end.char)
                for item in self.items
                if isinstance(item, CharacterRange)
            ]
        )
        self.options = frozenset(
            [item.char for item in self.items if isinstance(item, Char)]
        )

    def match(self, position: int, text: Sequence[T]):
        token = text[position]
        if token == "ε":
            raise RuntimeError()
        eq = token in self.options or any(
            start <= token <= stop for start, stop in self.intervals
        )
        eq_c = any(
            char_class.match(position, text)
            for char_class in filter(
                lambda item: isinstance(item, CharacterClass), self.items
            )
        )
        eq = eq or eq_c

        if self.negated:
            return not eq
        return eq

    def __eq__(self, other):
        if isinstance(other, CharacterGroup):
            return (
                self.options == other.options
                and self.intervals == other.intervals
                and self.negated == other.negated
            )
        return False

    def __repr__(self):
        return (
            "["
            + ("^" if self.negated else "")
            + "".join(f"{start}-{stop}" for start, stop in self.intervals)
            + "".join(self.options)
            + "".join(
                repr(char_class)
                for char_class in filter(
                    lambda item: isinstance(item, CharacterClass), self.items
                )
            )
            + "]"
        )

    def __lt__(self, other):
        return id(self) < id(other)

    def __hash__(self):
        return hash((self.options, self.intervals, self.negated))


@dataclass
class CharacterClass(RegexNode, CharacterGroupItem, CompoundMatchableMixin):
    item: str

    # to be filled later
    intervals: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    options: frozenset[str] = field(default_factory=set)
    # inverted = false
    negated: bool = False

    character_class2interval = {
        "w": ((("a", "z"), ("A", "Z")), frozenset(["_"]), False),
        "W": ((("a", "z"), ("A", "Z")), frozenset(["_"]), True),
        "d": ((("0", "9"),), frozenset(), False),
        "D": ((("0", "9"),), frozenset(), True),
        "s": ((), frozenset([" ", "\t", "\n", "\r", "\v", "\f"]), False),
        "S": ((), frozenset([" ", "\t", "\n", "\r", "\v", "\f"]), True),
    }

    def __post_init__(self):
        assert self.item in character_classes
        self.intervals, self.options, self.negated = self.character_class2interval[
            self.item
        ]

    def match(self, position: int, text: Sequence[T]):
        token = text[position]
        if token == "ε":
            raise RuntimeError()
        eq = token in self.options or any(
            start <= token <= stop for start, stop in self.intervals
        )
        if self.negated:
            return not eq
        return eq

    def __eq__(self, other):
        if isinstance(other, CharacterClass):
            return (
                self.options == other.options
                and self.intervals == other.intervals
                and self.negated == other.negated
            )
        return False

    def to_fsm(
        self, transitions: MutableMapping[State, TransitionsProvider]
    ) -> StatePair:
        return trivial(self, transitions)

    def __hash__(self):
        return hash((self.negated, self.options, self.intervals))

    def __repr__(self):
        return f"\\\\{self.item}"


@dataclass
class CharacterRange(CharacterGroupItem):
    start: Char
    end: Optional[Char]


class AnchorType(Enum):
    StartOfString = "^"
    EndOfString = "$"
    # must be escaped
    AnchorWordBoundary = "b"
    AnchorNonWordBoundary = "B"
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
                return AnchorType.AnchorWordBoundary
            case "B":
                return AnchorType.AnchorNonWordBoundary
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


def is_word_boundary(text: Sequence[T], position: int) -> bool:
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
class Anchor(SubExpressionItem, CompoundMatchableMixin):
    anchor_type: AnchorType

    def to_fsm(
        self, transitions: MutableMapping[State, TransitionsProvider]
    ) -> StatePair:
        return trivial(self, transitions)

    def match(self, position: int, text: Sequence[T]) -> bool:
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
            case AnchorType.AnchorWordBoundary:
                return is_word_boundary(text, position)
            case AnchorType.AnchorNonWordBoundary:
                return not is_word_boundary(text, position)

        raise NotImplementedError

    def __hash__(self):
        return hash(self.anchor_type)

    def __repr__(self):
        return self.anchor_type.name


class RegexParser:
    def __init__(self, regex: str):
        self._regex = regex.replace(" ", "").replace("\t", "")
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
        assert self.current() == char, f"expected {char} got {self.current()}"
        self._pos += 1

    def consume_and_return(self):
        char = self.current()
        self.consume(char)
        return char

    def optional(self, expected):
        if self._pos < len(self._regex) and self.current() == expected:
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
        return self._pos < len(self._regex) and (
            self.current() == "."
            or self.can_parse_character_class_or_group()
            or self.can_parse_char()
            or self.can_parse_escaped()
        )

    def can_parse_sub_expression_item(self):
        return self._pos < len(self._regex) and (
            self.can_parse_group() or self.can_parse_anchor() or self.can_parse_match()
        )

    def matches(self, char):
        return self._pos < len(self._regex) and self.current() == char

    def matches_any(self, options):
        return self._pos < len(self._regex) and self.current() in options

    def parse_expression(self):
        # Expression ::= Subexpression ("|" Expression)?
        pos = self._pos
        sub_exprs = self.parse_sub_expression()
        expr = None
        if self.matches("|"):
            self.consume("|")
            expr = self.parse_expression()
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

    def parse_group(self):
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
                return quantifier.item.expand(expr)
        return Group(self._pos, expr, quantifier, is_capturing)

    def can_parse_quantifier(self):
        return self._pos < len(self._regex) and self.current() in ("*", "+", "?", "{")

    def parse_quantifier(self):
        if self.current() in ("*", "+", "?"):
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
                return quantifier.item.expand(match_item)
        return Match(pos, match_item, quantifier)

    def can_parse_character_group(self):
        return self.current() == "["

    def parse_character_class(self) -> CharacterClass:
        self.consume("\\")
        return CharacterClass(self._pos, self.consume_and_return())

    def parse_character_range(self, char: Char) -> CharacterRange:
        self.consume("-")
        to = self.parse_char()
        assert to.char != "]"
        return CharacterRange(char, to)

    def can_parse_character_class(self):
        return (
            self.matches("\\")
            and self._pos + 1 < len(self._regex)
            and self.current(1) in character_classes
        )

    def parse_character_group_item(self) -> CharacterGroupItem | Char:
        if self.matches("\\") and self.current(1) in character_classes:
            return self.parse_character_class()
        else:
            char = self.parse_char()
            if self.matches("-"):
                return self.parse_character_range(char)
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
        while self.can_parse_char() and self.current() != "]" or self.current() == "\\":
            items.append(self.parse_character_group_item())
        self.consume("]")
        return CharacterGroup(group_pos, items, negated)

    def parse_char(self):
        if self.can_parse_escaped():
            return self.parse_escaped()
        assert self.can_parse_char()
        return Char(self._pos - 1, self.consume_and_return())

    def can_parse_escaped(self):
        return (
            self._pos < len(self._regex)
            and self.current() == "\\"
            and self.current(1) in ESCAPED
        )

    def can_parse_anchor(self):
        return self._pos < len(self._regex) and (
            (
                self.current() == "\\"
                and self.current(1) in {"A", "z", "Z", "G", "b", "B"}
            )
            or self.current() in ("^", "$")
        )

    def parse_escaped(self):
        self.consume("\\")
        return Char(self._pos - 1, self.consume_and_return())

    def can_parse_character_class_or_group(self):
        return self.can_parse_character_class() or self.can_parse_character_group()

    def parse_character_class_or_group(self):
        if self.can_parse_character_class():
            return self.parse_character_class()
        else:
            return self.parse_character_group()

    def parse_match_item(self):
        if self.matches("."):
            # parse AnyCharacter
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


if __name__ == "__main__":
    # p = Parser(r"(ab){3, 4}[^A-Za0-9_]+(\w)*")
    # p = Parser(r"(ab)|(df)")
    # p = RegexParser(r"a*b+a.a*b|d+[A-Z]?")
    # t = defaultdict(lambda: SymbolDispatchedMapping(set))
    # pprint(p._start_state)
    # initial, accept = p._start_state.to_fsm(t)
    #
    # # symbols
    # symbols = set()
    # states = set()
    # for s1 in t:
    #     states.add(s1)
    #     for sym, s2 in t[s1].items():
    #         states = states | s2
    #         symbols.add(sym)
    # symbols.discard(Epsilon)
    #
    # dfa1 = DFA(nfa=NFA(t, states, symbols, initial, accept))
    # dfa1.minimize()
    # dfa1.draw_with_graphviz()
    ...
