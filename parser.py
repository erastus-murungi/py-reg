import re
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntFlag, auto
from sys import maxsize
from typing import Generic, Hashable, Optional, TypeVar

T = TypeVar("T")


class RegexFlag(IntFlag):
    NOFLAG = auto()
    IGNORECASE = auto()
    MULTILINE = auto()
    DOTALL = auto()  # make dot match newline
    FREESPACING = auto()


class InvalidCharacterRange(Exception):
    ...


INLINE_MODIFIER_START = "(?"

pattern = re.compile(r"(?<!^)(?=[A-Z])")


@dataclass
class RegexNode(ABC):
    pos: int = field(repr=False)

    def accept(self, visitor: "RegexpNodesVisitor"):
        method_name = f"visit_{pattern.sub('_', self.__class__.__name__).lower()}"
        visit = getattr(visitor, method_name)
        return visit(self)

    @abstractmethod
    def string(self) -> str:
        ...


class AnchorType(Enum):
    StartOfString = "^"
    EndOfString = "$"
    EmptyString = ""

    # must be escaped
    WordBoundary = "\\b"
    NonWordBoundary = "\\B"
    StartOfStringOnly = "\\A"
    EndOfStringOnlyNotNewline = "\\z"
    EndOfStringOnlyMaybeNewLine = "\\Z"

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
                return AnchorType.StartOfStringOnly
            case "z":
                return AnchorType.EndOfStringOnlyNotNewline
            case "Z":
                return AnchorType.EndOfStringOnlyMaybeNewLine
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


class Matchable(Hashable):
    @abstractmethod
    def match(self, text: str, position: int, flags: RegexFlag) -> bool:
        ...

    def is_opening_group(self):
        return isinstance(self, Tag) and self.tag_type == TagType.GroupEntry

    def is_closing_group(self):
        return isinstance(self, Tag) and self.tag_type == TagType.GroupExit

    def increment(self, index: int) -> int:
        """We keep the index the same only when we are a `Virtual` node"""
        return index + (not isinstance(self, Virtual))


class Virtual(Matchable, ABC):
    ...


class TagType(Enum):
    Epsilon = "ε"
    GroupEntry = "GroupEntry"
    GroupExit = "GroupExit"
    GroupLink = ""
    Fence = "Fence"


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


class SubExpressionItem(RegexNode, ABC):
    pass


@dataclass(slots=True)
class Anchor(SubExpressionItem, Virtual):
    anchor_type: AnchorType

    @staticmethod
    def empty_string(pos: int = maxsize) -> "Anchor":
        return Anchor(pos, AnchorType.EmptyString)

    def match(self, text: str, position: int, flags: RegexFlag) -> bool:
        match self.anchor_type:
            case AnchorType.StartOfString:
                # match the start of the string
                return position == 0 or (
                    # and in MULTILINE mode also matches immediately after each newline.
                    (RegexFlag.MULTILINE & flags)
                    and (not text or (position > 0) and text[position - 1] == "\n")
                )
            case AnchorType.EndOfString:
                # . foo matches both ‘foo’ and ‘foobar’,
                # while the regular expression foo$ matches only ‘foo’.
                # More interestingly, searching for foo.$ in 'foo1\nfoo2\n' matches ‘foo2’ normally,
                # but ‘foo1’ in MULTILINE mode; searching for a single $ in 'foo\n' will find two (empty) matches:
                # one just before the newline, and one at the end of the string.

                return (
                    position >= len(text)  # Matches the end of the string
                    or position
                    == len(text)
                    - 1  # or just before the newline at the end of the string
                    and text[position] == "\n"
                ) or (
                    # and in MULTILINE mode also matches before a newline
                    RegexFlag.MULTILINE & flags
                    and (position < len(text) and text[position] == "\n")
                )
            case AnchorType.StartOfStringOnly:
                # matches only at the start of the string
                return position == 0
            case AnchorType.EndOfStringOnlyMaybeNewLine:
                return (
                    position == len(text) - 1 and text[position] == "\n"
                ) or position >= len(text)
            case AnchorType.EndOfStringOnlyNotNewline:
                return position >= len(text)
            case AnchorType.WordBoundary:
                return is_word_boundary(text, position)
            case AnchorType.NonWordBoundary:
                return text and not is_word_boundary(text, position)
            case AnchorType.EmptyString:
                # empty string always matches
                return True

        raise NotImplementedError

    def string(self):
        return self.anchor_type.value

    def __hash__(self):
        return hash(self.anchor_type)

    def __repr__(self):
        return self.anchor_type.name


@dataclass
class Expression(RegexNode):
    seq: list[SubExpressionItem]
    alternate: Optional["Expression"] = None

    def string(self):
        seq = "".join(item.string() for item in self.seq)
        if self.alternate is not None:
            return f"{seq}|{self.alternate.string()}"
        return seq


ESCAPED = set(". \\ + * ? [ ^ ] $ ( ) { } = < > | -".split())

CHARACTER_CLASSES = {"w", "W", "s", "S", "d", "D"}

UNESCAPED_IN_CHAR_GROUP = ESCAPED - {"]"}


@dataclass
class Quantifier:
    lazy: bool
    char: Optional[str]
    range_quantifier: Optional[tuple[int, Optional[int]]] = None

    def validate_range(self):
        assert self.range_quantifier is not None
        start, end = self.range_quantifier
        if end is None:
            if start < 0:
                raise InvalidCharacterRange(
                    f"fixed quantifier, {{n}} must be >= 0: not {start}"
                )
        elif isinstance(end, int):
            if end == maxsize:
                if start < 0:
                    raise InvalidCharacterRange(
                        f"for {{n,}} quantifier, {{n}} must be >= 0: not {start}"
                    )
            else:
                if start < 0:
                    raise InvalidCharacterRange(
                        f"for {{n, m}} quantifier, {{n}} must be >= 0: not {start}"
                    )
                if end < start:
                    raise InvalidCharacterRange(
                        f"for {{n, m}} quantifier, {{m}} must be >= {{n}}: not {end}"
                    )
        elif start == 0:
            if not isinstance(end, int):
                raise InvalidCharacterRange(f"invalid upper bound {end}")
            if end < 1:
                raise InvalidCharacterRange(
                    f"for {{, m}} quantifier, {{m}} must be >= 1: not {end}"
                )
        else:
            raise InvalidCharacterRange(f"invalid range {{{start}, {end}}}")

    def __post_init__(self):
        if self.char is not None:
            assert self.char in ("?", "+", "*"), f"invalid quantifier {self.char}"
        else:
            assert self.range_quantifier is not None
            self.validate_range()

    def string(self):
        # TODO: Fix string method of range quantifier so that it appears as it was in original regex
        if self.char is not None:
            base = self.char
        else:
            base = f"{{{self.range_quantifier}}}"
        return base + "?" if self.lazy else ""


@dataclass
class Group(SubExpressionItem):
    expression: Expression
    quantifier: Optional[Quantifier]
    group_index: Optional[int]
    substr: Optional[str]

    def capturing(self):
        return self.group_index is not None

    def string(self):
        expression = f"({'' if self.capturing() else '?:'}{self.expression.string()})"
        if self.quantifier is not None:
            return f"{expression}{self.quantifier.string()}"
        return expression


class MatchItem(SubExpressionItem, ABC):
    pass


@dataclass
class Match(SubExpressionItem):
    item: MatchItem
    quantifier: Optional[Quantifier]

    def string(self):
        return (
            f'{self.item.string()}{self.quantifier.string() if self.quantifier else ""}'
        )


@dataclass
class MatchAnyCharacter(MatchItem, Matchable):
    ignore: tuple = ("\n",)

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
        return f"{self.start}-{self.end}"


class RegexpNodesVisitor(Generic[T], metaclass=ABCMeta):
    @abstractmethod
    def visit_anchor(self, anchor: Anchor) -> T:
        ...

    @abstractmethod
    def visit_expression(self, expression: Expression) -> T:
        ...

    @abstractmethod
    def visit_group(self, group: Group) -> T:
        ...

    @abstractmethod
    def visit_match(self, group: Group) -> T:
        ...

    @abstractmethod
    def visit_match_any_character(self, meta_char: MatchAnyCharacter) -> T:
        ...

    @abstractmethod
    def visit_character_scalar(self, character_scalar: CharacterScalar) -> T:
        ...

    @abstractmethod
    def visit_character_group(self, character_group: CharacterGroup) -> T:
        ...


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
        if self.can_parse_group():
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
        return Group(self._pos, expr, quantifier, index, self._regex[start:end])

    def can_parse_quantifier(self):
        return self.matches_any(("*", "+", "?", "{"))

    def parse_quantifier(self) -> Quantifier:
        if self.matches_any(("*", "+", "?")):
            quantifier_char = self.consume_and_return()
            return Quantifier(self.optional("?"), quantifier_char)
        else:
            quantifier_range = self.parse_range_quantifier()
            return Quantifier(self.optional("?"), None, quantifier_range)

    def parse_int(self):
        digits = []
        while self.current().isdigit():
            digits.append(self.consume_and_return())
        return int("".join(digits))

    def parse_range_quantifier(self) -> tuple[int, Optional[int]]:
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
        return lower, upper

    def parse_match(self) -> Match | Expression:
        # Match ::= MatchItem Quantifier?
        pos = self._pos
        match_item = self.parse_match_item()
        quantifier = None
        if self.can_parse_quantifier():
            quantifier = self.parse_quantifier()
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
                self._pos, (CharacterRange("0", "9"),), self.matches("D")
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
            while (
                self.can_parse_char()
                or self.matches("\\")
                or self.matches_any(UNESCAPED_IN_CHAR_GROUP)
            ):
                items.append(self.parse_character_group_item())
                state = self.save_state()
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
