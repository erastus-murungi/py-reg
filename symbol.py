import reprlib
from abc import ABC, abstractmethod
from typing import Final, Generic, TypeVar

from utils import Comparable

T = TypeVar("T", bound=Comparable)


class Symbol(Generic[T], ABC):
    @abstractmethod
    def match(self, token: T) -> bool:
        ...


class Character(Symbol):
    def __init__(self, char: T):
        self.char = char

    def match(self, token) -> bool:
        return self.char == token

    def __eq__(self, other) -> bool:
        return other == self.char

    def __repr__(self):
        return f"{self.char}"

    def __hash__(self):
        return hash(self.char)


class Operator(Character):
    def __init__(self, char: T):
        super().__init__(char)

    def __repr__(self):
        return f"Op({self.char})"


class CompoundSymbol(Symbol):
    @abstractmethod
    def match(self, other) -> bool:
        pass


class MetaSequence(CompoundSymbol):
    def __init__(self, char: T):
        self.char = char

    def match(self, token: T) -> bool:
        raise NotImplementedError

    def __repr__(self):
        return f"Meta(\\{self.char})"


class OneOf(CompoundSymbol):
    def __init__(self, options: set[T], source="", negated=False):
        self.options = frozenset(options)
        self.source = source
        self.negated = negated

    def match(self, other):
        if isinstance(other, OneOf):
            return self.options == other.options
        return False

    def __eq__(self, token):
        if self.negated:
            return token not in self.options
        return token in self.options

    def __repr__(self):
        if self.source:
            return self.source
        return reprlib.repr(self.options)

    def __lt__(self, other):
        return id(self) < id(other)

    def __hash__(self):
        return hash(self.options)


class AnyCharacter(CompoundSymbol):
    def __init__(self, ignore=("ε",)):
        self.ignore = ignore

    def __eq__(self, token):
        return token not in self.ignore

    def match(self, other):
        return isinstance(other, AnyCharacter) and other.ignore == self.ignore

    def __repr__(self):
        return "Any"

    def __hash__(self):
        return hash(".") ^ 12934


ESCAPED = set(". \\ + * ? [ ^ ] $ ( ) { } = ! < > | : -".split())

OpeningParen = Operator("(")
ClosingParen = Operator(")")
KleeneClosure = Operator("*")
OneOrMore = Operator("+")
Alternation = Operator("|")
Concatenate = Operator(".")
ZeroOrOne = Operator("?")
Caret = Operator("^")
OpeningBrace = Operator("{")
ClosingBrace = Operator("}")

AllOps = {
    Alternation,
    ZeroOrOne,
    OneOrMore,
    KleeneClosure,
    Caret,
    OpeningParen,
    ClosingParen,
}

BinOps = {Caret, Alternation}

Epsilon = Character("ε")

PRECEDENCE: Final[dict[Operator, int]] = {
    OpeningParen: 1,
    Alternation: 2,
    Concatenate: 3,  # explicit concatenation operator
    ZeroOrOne: 4,
    KleeneClosure: 4,
    OneOrMore: 4,
    Caret: 5,
}


def precedence(token) -> int:
    try:
        return PRECEDENCE[token]
    except KeyError:
        return 6


def gen_symbols_exclude_precedence_ops(postfix_regexp: list[Symbol]) -> set[Symbol]:
    symbols = set(postfix_regexp) - set(PRECEDENCE.keys())
    return symbols
