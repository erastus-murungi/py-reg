import sys
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass, field
from enum import Enum
from math import inf
from typing import MutableMapping, Optional, Sequence

from core import (CompoundMatchableMixin, MatchableMixin, State, T,
                  TransitionsProvider)

ESCAPED = set(". \\ + * ? [ ^ ] $ ( ) { } = ! < > | -".split())

anchors = {"$", "^"}

character_classes = {"w", "W", "s", "S", "d", "D"}

StatePair = tuple[State, State]

no_escape_in_group = {"\\", "-", "[", ":", ".", ">"}


def zero_or_more(state_pair, transitions, lazy) -> StatePair:
    source, sink = state_pair
    new_start, new_accept = State.pair()
    transitions[sink][Epsilon].add(source)
    transitions[new_start][Epsilon].add(source)
    transitions[sink][Epsilon].add(new_accept)
    transitions[new_start][Epsilon].add(new_accept)
    s1_start, s2_end = trivial(Anchor.empty_string(), transitions)
    concatenate(s2_end, new_start, transitions)
    source.lazy = sink.lazy = lazy
    return s1_start, new_accept


def one_or_more(state_pair: StatePair, transitions, lazy: bool) -> StatePair:
    source, sink = state_pair
    transitions[sink][Epsilon].add(source)
    source.lazy = sink.lazy = lazy
    return source, sink


def zero_or_one(state_pair: StatePair, transitions, lazy: bool) -> StatePair:
    source, sink = state_pair
    transitions[source][Epsilon].add(sink)
    s1_start, s2_end = trivial(Anchor.empty_string(), transitions)
    concatenate(s2_end, source, transitions)
    source.lazy = sink.lazy = lazy
    return s1_start, sink


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
    def fsm(self, transitions: MutableMapping[State, TransitionsProvider]) -> StatePair:
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
                    return one_or_more(state_pair, transitions, self.lazy)
                case QuantifierType.ZeroOrMore:
                    return zero_or_more(state_pair, transitions, self.lazy)
                case QuantifierType.ZeroOrOne:
                    return zero_or_one(state_pair, transitions, self.lazy)
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

    def expand(self, item: "SubExpressionItem", lazy: bool):
        # e{3} expands to eee; e{3,5} expands to eeee?e?, and e{3,} expands to eee+.

        seq = []
        for _ in range(self.start):
            seq.append(copy(item))

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

    def fsm(self, transitions) -> StatePair:
        nodes = [subexpression.fsm(transitions) for subexpression in self.seq]
        for (_, s1_accept), (s2_start, _) in zip(nodes, nodes[1:]):
            transitions[s1_accept][Epsilon].add(s2_start)
        state_pair = nodes[0][0], nodes[-1][1]
        if self.alternate is None:
            return state_pair
        return alternation(state_pair, self.alternate.fsm(transitions), transitions)


@dataclass
class Group(SubExpressionItem):
    expression: Expression
    quantifier: Optional[Quantifier]
    is_capturing: bool = False

    def fsm(self, transitions: MutableMapping[State, TransitionsProvider]) -> StatePair:
        state_pair = self.expression.fsm(transitions)
        if self.quantifier:
            return self.quantifier.transform(state_pair, transitions)
        return state_pair


class MatchItem(SubExpressionItem, ABC):
    pass


@dataclass
class Match(SubExpressionItem):
    item: MatchItem
    quantifier: Optional[Quantifier]

    def fsm(self, transitions: dict[State, TransitionsProvider]) -> StatePair:
        state_pair = self.item.fsm(transitions)
        if self.quantifier is None:
            return state_pair
        return self.quantifier.transform(state_pair, transitions)


@dataclass
class MatchAnyCharacter(MatchItem, CompoundMatchableMixin):
    ignore: tuple = ("ε", "\n")

    def fsm(self, transitions: dict[State, TransitionsProvider]) -> StatePair:
        return trivial(self, transitions)

    def __eq__(self, other):
        return isinstance(other, MatchAnyCharacter) and other.ignore == self.ignore

    def match(self, text, position, flags) -> bool:
        return position < len(text) and text[position] not in self.ignore

    def __repr__(self):
        return "Any"

    def __hash__(self):
        return hash(".") ^ 12934


class CharacterGroupItem(MatchableMixin, ABC):
    pass


@dataclass
class Char(CharacterGroupItem, MatchItem):
    char: str

    def fsm(self, transitions) -> tuple[State, State]:
        source, sink = State.pair()
        transitions[source][self].add(sink)
        return source, sink

    def match(self, text, position, flags) -> bool:
        if position < len(text):
            return self.char == text[position]
        return False

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


@dataclass
class CharacterGroup(MatchCharacterClass, CompoundMatchableMixin):
    items: tuple[CharacterGroupItem, ...]
    negated: bool = False

    def fsm(self, transitions: dict[State, TransitionsProvider]) -> tuple[State, State]:
        source, sink = State.pair()
        transitions[source][self].add(sink)
        return source, sink

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
        return f"{self.items}"

    def __lt__(self, other):
        return id(self) < id(other)

    def __hash__(self):
        return hash((self.items, self.negated))


@dataclass
class CharacterRange(CharacterGroupItem, CompoundMatchableMixin):
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

    def fsm(self, transitions: MutableMapping[State, TransitionsProvider]) -> StatePair:
        return trivial(self, transitions)

    @staticmethod
    def empty_string(pos=sys.maxsize):
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

    def matches_any(self, options, lookahead: int = 0):
        return (
            self._pos + lookahead < len(self._regex)
            and self.current(lookahead) in options
        )

    def parse_expression(self) -> Expression:
        # Expression ::= Subexpression ("|" Expression)?
        pos = self._pos
        sub_exprs = self.parse_sub_expression()
        expr = None
        if self.matches("|"):
            self.consume("|")
            if self.can_parse_sub_expression_item():
                expr = self.parse_expression()
            else:
                expr = Anchor.empty_string(self._pos)
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
                return quantifier.item.expand(match_item, quantifier.lazy)
        return Match(pos, match_item, quantifier)

    def can_parse_character_group(self):
        return self.current() == "["

    def parse_character_class(self) -> CharacterGroup:
        self.consume("\\")
        if self.matches_any(("w", "W")):
            return CharacterGroup(
                self._pos,
                (
                    CharacterRange("A", "Z"),
                    CharacterRange("a", "z"),
                    Char(self._pos, "_"),
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
                        lambda c: Char(self._pos, c),
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
                return Char(self._pos, self.consume_and_return())
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
        return Char(self._pos - 1, self.consume_and_return())

    def can_parse_escaped(self):
        return (
            self._pos < len(self._regex)
            and self.matches("\\")
            and self.matches_any(ESCAPED, 1)
        )

    def can_parse_anchor(self):
        return self._pos < len(self._regex) and (
            (self.matches("\\") and self.matches_any({"A", "z", "Z", "G", "b", "B"}, 1))
            or self.matches_any(("^", "$"))
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
