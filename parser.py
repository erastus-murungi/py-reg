from abc import ABC
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional
from enum import Enum

from symbol import ESCAPED

anchors = {"$"}

character_classes = {"w", "W", "s", "S", "d", "D"}


class MatchNotFound(ValueError):
    pass


@dataclass
class RegexNode(ABC):
    pos: int = field(repr=False)


class Operator(RegexNode, ABC):
    pass


class QuantifierItem(RegexNode):
    pass


@dataclass
class Quantifier(Operator):
    item: QuantifierItem
    lazy: bool = False


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
class RangeQuantifier(QuantifierItem):
    start: int
    end: Optional[int] = None


class SubExpressionItem(RegexNode, ABC):
    pass


class SubExpression(RegexNode):
    items: list[SubExpressionItem]


@dataclass
class Expression:
    sub_expr: list[SubExpression]
    expr: Optional["Expression"]


@dataclass
class Group(SubExpressionItem):
    is_capturing: bool
    expression: Expression
    quantifier: Optional[Quantifier]


class MatchItem(SubExpressionItem, ABC):
    pass


@dataclass
class Match:
    item: MatchItem
    quantifier: Quantifier


@dataclass
class MatchAnyCharacter(MatchItem):
    ignore: tuple = ()


@dataclass
class Char(MatchItem):
    char: str


class MatchCharacterClass(MatchItem, ABC):
    pass


class CharacterGroupItem(ABC):
    pass


@dataclass
class CharacterGroup(MatchCharacterClass):
    items: list[CharacterGroupItem]
    negated: bool = False


@dataclass
class CharacterClass(CharacterGroupItem):
    item: str

    def __post_init__(self):
        assert self.item in character_classes


@dataclass
class CharacterRange(CharacterGroupItem):
    start: Char
    end: Optional[Char]


class Parser:
    def __init__(self, regex: str):
        self.regex = regex.replace(" ", "").replace("\t", "")
        self.pos = 0
        self.root = self.parse_regex()
        if self.pos < len(self.regex):
            raise ValueError(
                f"could not finish parsing regex, left = {self.regex[self.pos:]}"
            )

    def consume(self, char):
        assert self.current() == char, f"expected {char} got {self.current()}"
        self.pos += 1

    def consume_and_return(self):
        char = self.current()
        self.consume(char)
        return char

    def optional(self, expected):
        if self.pos < len(self.regex) and self.current() == expected:
            self.consume(expected)
            return True
        return False

    def current(self, lookahead=None):
        if lookahead is not None:
            return self.regex[self.pos + lookahead]
        return self.regex[self.pos]

    def parse_regex(self):
        if self.current() == "^":
            self.consume("^")
        return self.parse_expression()

    def can_parse_group(self):
        return self.current() == "("

    def can_parse_anchor(self):
        return self.current() in anchors

    def can_parse_char(self):
        return self.pos < len(self.regex) and self.current() not in ESCAPED

    def can_parse_match(self):
        return self.pos < len(self.regex) and (
            self.current() == "."
            or self.can_parse_character_class_or_group()
            or self.can_parse_char()
            or self.can_parse_escaped()
        )

    def can_parse_sub_expression_item(self):
        return self.pos < len(self.regex) and (
            self.can_parse_group() or self.can_parse_anchor() or self.can_parse_match()
        )

    def matches(self, char):
        return self.pos < len(self.regex) and self.current() == char

    def parse_expression(self):
        # Expression ::= Subexpression ("|" Expression)?
        sub_exprs = self.parse_sub_expression()
        expr = None
        if self.matches("|"):
            expr = self.parse_expression()
        return Expression(sub_exprs, expr)

    def parse_sub_expression(self) -> list[SubExpression]:
        # Subexpression ::= SubexpressionItem+
        sub_exprs = [self.parse_sub_expression_item()]
        while self.can_parse_sub_expression_item():
            sub_exprs.append(self.parse_sub_expression_item())
        return sub_exprs

    def parse_sub_expression_item(self):
        if self.current() == "(":
            return self.parse_group()
        elif self.current() in anchors:
            return self.parse_anchor()
        else:
            return self.parse_match()

    def parse_group(self):
        self.consume("(")
        is_capturing = True
        if self.regex.startswith("?:"):
            self.consume("?:")
            is_capturing = False
        expr = self.parse_expression()
        self.consume(")")
        quantifier = None
        if self.can_parse_quantifier():
            quantifier = self.parse_quantifier()
        return Group(self.pos, is_capturing, expr, quantifier)

    def can_parse_quantifier(self):
        return self.pos < len(self.regex) and self.current() in ("*", "+", "?", "{")

    def parse_quantifier(self):
        if self.current() in ("*", "+", "?"):
            quantifier_item = QuantifierChar(
                self.pos, QuantifierType.get(self.consume_and_return())
            )
        else:
            quantifier_item = self.parse_range_quantifier()
        return Quantifier(self.pos, quantifier_item, self.optional("?"))

    def parse_int(self):
        digits = []
        while self.current().isdigit():
            digits.append(self.consume_and_return())
        return int("".join(digits))

    def parse_range_quantifier(self) -> RangeQuantifier:
        # RangeQuantifier ::= "{" RangeQuantifierLowerBound ( "," RangeQuantifierUpperBound? )? "}"
        self.consume("{")
        # RangeQuantifierLowerBound = Integer
        lower_bound = self.parse_int()
        upper_bound = None
        while self.current() == ",":
            self.consume_and_return()
            if self.current().isdigit():
                upper_bound = self.parse_int()
        self.consume("}")
        return RangeQuantifier(lower_bound, upper_bound)

    def parse_match(self):
        # Match ::= MatchItem Quantifier?
        match_item = self.parse_match_item()
        quantifier = None
        if self.can_parse_quantifier():
            quantifier = self.parse_quantifier()
        return Match(match_item, quantifier)

    def can_parse_character_group(self):
        return self.current() == "["

    def parse_character_class(self):
        self.consume("\\")
        return CharacterClass(self.consume_and_return())

    def parse_character_range(self, char: Char) -> CharacterRange:
        self.consume("-")
        to = self.parse_char()
        assert to.char != "]"
        return CharacterRange(char, to)

    def can_parse_character_class(self):
        return (
            self.matches("\\")
            and self.pos + 1 < len(self.regex)
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
        group_pos = self.pos
        while self.can_parse_char() and self.current() != "]" or self.current() == "\\":
            items.append(self.parse_character_group_item())
        self.consume("]")
        return CharacterGroup(group_pos, items, negated)

    def parse_char(self):
        assert self.can_parse_char()
        if self.can_parse_escaped():
            return self.parse_escaped()
        return Char(self.pos, self.consume_and_return())

    def can_parse_escaped(self):
        return (
            self.pos < len(self.regex)
            and self.current() == "\\"
            and self.current(1) in ESCAPED
        )

    def parse_escaped(self):
        self.consume("\\")
        return Char(self.pos, self.consume_and_return())

    def can_parse_character_class_or_group(self):
        return self.can_parse_character_class() or self.can_parse_character_group()

    def parse_character_class_or_group(self):
        if self.can_parse_character_class():
            return self.parse_character_class()
        else:
            return self.parse_character_group()

    def parse_match_item(self):
        if self.current() == ".":
            # parse AnyCharacter
            return MatchAnyCharacter(self.pos)
        elif self.can_parse_character_class_or_group():
            return self.parse_character_class_or_group()
        else:
            return self.parse_char()

    def parse_anchor(self):
        raise NotImplementedError

    def __repr__(self):
        return f"Parser({self.regex})"


if __name__ == "__main__":
    p = Parser(r"(ab){3, 4}[^A-Za0-9_]+(\w)*")
    pprint(p.root)
