import re
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from sys import maxsize
from typing import Final, Generic, Hashable, Optional, TypeVar

from reg.matcher import Context, Cursor
from reg.utils import RegexFlag

V = TypeVar("V")

INLINE_MODIFIER_START = "(?"
ESCAPED = set(". \\ + * ? [ ^ ] $ ( ) { } = < > | -".split())
CHARACTER_CLASSES = {"w", "W", "s", "S", "d", "D"}
ANCHORS = {"A", "z", "Z", "G", "b", "B"}
UNESCAPED_IN_CHAR_GROUP = ESCAPED - {"]"}
QUANTIFIER_OPTIONS = ("?", "+", "*")


class RegexpParsingError(Exception):
    ...


def is_word_character(char: str) -> bool:
    """
    Check if `char` is just a single alphabetic character or an underscore

    Parameters
    ----------
    char: str
        The character to check

    Returns
    -------
        True if `char` is just a single alphabetic character or an underscore
        False otherwise
    Examples
    --------
    >>> is_word_character('a')
    True
    >>> is_word_character('1')
    False
    >>> is_word_character('_')
    True
    >>> is_word_character('abc')
    False
    >>> is_word_character('')
    False
    """
    return len(char) == 1 and char.isalpha() or char == "_"


def is_word_boundary(text: str, position: int) -> bool:
    """
    Check if text[position:] is a word boundary

    There are three different positions that qualify as word boundaries:

        1. Before the first character in the string, if the first character is a word character.
        2. After the last character in the string, if the last character is a word character.
        3. Between two characters in the string, where one is a word character and
            the other is not a word character.

    Parameters
    ---------
    text: str
        a text to check for the word boundary
    position: int
        the position in the text to check the word boundary

    Returns
    -------
    bool
        True is text[position:] is a word boundary and False otherwise

    Raises
    ------
    ValueError
        If given text is the empty string


    Examples
    --------
    >>> txt = 'abc'
    >>> is_word_boundary(txt, 0)
    True
    >>> is_word_boundary(txt, 1)
    False
    >>> is_word_boundary('1abc', 0)  # first letter not a word character
    True
    >>> is_word_boundary('', 0)  # empty string
    Traceback (most recent call last):
        ...
    ValueError: expected a non-empty string
    >>> is_word_boundary('ab', 2)  # past last letter
    True
    >>> is_word_boundary('ab', 1)  # at last letter
    False
    >>> is_word_boundary('a c', 0)
    True
    >>> is_word_boundary('a c', 1)
    True
    >>> is_word_boundary('a c', 2)
    True
    >>> is_word_boundary('abc', 1)
    False
    """
    if len(text) == 0:
        raise ValueError("expected a non-empty string")

    case1 = position == 0 and is_word_character(text[position])
    case2 = position == len(text) and is_word_character(text[position - 1])
    case3 = position < len(text) and (
        is_word_character(text[position - 1]) ^ is_word_character(text[position])
    )
    return case1 or case2 or case3


class Matcher(Hashable):
    """
    Base class for all matching functions

    Notes
    -----
    If a matcher accepts, then we can proceed in one or both the text and the pattern
    There are essentially two types of matching functions, those that consume characters and those
    that don't, but instead consume a conditon, eg a word boundary

    We override the call method so that a Matcher class instance can be called just like a function
    E.g x(cursor, context)

    Examples
    --------
    >>> Matcher()
    Traceback (most recent call last):
        ...
    TypeError: Can't instantiate abstract class Matcher with abstract methods __call__, __hash__

    """

    @abstractmethod
    def __call__(self, cursor: Cursor, context: Context) -> bool:
        """
        Check if this matcher accepts the cursor and context given

        Examples
        --------
        >>> character_matcher = Character('a')
        >>> cs = (1, [])
        >>> ctx = Context('baba', RegexFlag.NOFLAG)
        >>> character_matcher(cs, ctx)
        True
        """
        ...

    def update(self, cursor: Cursor) -> Cursor:
        """
        Update the index and groups of a cursor object
        It is assumed that the cursor has already been accepted by this matcher

        Parameters
        ----------
        cursor: Cursor
            A cursor to update

        Returns
        -------
        Cursor
            A new cursor with updated parameters

        Examples
        --------
        >>> cs = (0, [])
        >>> character_matcher = Character('a')
        >>> character_matcher.update(cs)
        Cursor(position=1, groups=[])
        >>> from sys import maxsize
        >>> cs = (1, [-1, -1])
        >>> group_entry = Anchor.group_entry(0)
        >>> group_entry.update(cs)
        Cursor(position=1, groups=[1, -1])

        Notes
        -----
        We have to create a new cursor object because cursor objects are immutable
        We update groups only if this object is a group anchor
        Else, we just update the cursor position and pass in the same groups object to
        the new cursor

        """
        if isinstance(self, Anchor) and (
            self.anchor_type == AnchorType.GroupEntry
            or self.anchor_type == AnchorType.GroupExit
        ):
            return self._update_groups_and_index(cursor)
        return self.update_index(cursor)

    def update_index(self, cursor: Cursor) -> Cursor:
        """
        Only increment the position of the cursor
        Pass in the groups object as is
        """
        position, groups = cursor
        return Cursor(self.increment_index(position), groups)

    def increment_index(self, index: int) -> int:
        """
        Depending on the type of the matcher, increment the index to a new value
        We keep the index the same only when we are an Anchor

        Parameters
        ----------
        index: int
            The index to increment

        Notes
        -----
        As an optimization, if we have a multi-character matcher, then we have to update this method
        to return the length of the multi-character match

        """
        return index + (not isinstance(self, Anchor))


@dataclass
class RegexNode(ABC):
    # finds upper case letters which are not at the beginning of a string
    pattern = re.compile(r"(?<!^)(?=[A-Z])")

    def accept(self, visitor: "RegexNodesVisitor"):
        """
        This is the acceptor of an instance of RegexNodesVisitor
        Works by finding the appropriate method in the visitor.

        The appropriate visit method for a class X is `visit_ + to_camel_case(X)`

        Examples
        --------
        >>> PrintNode = type(
        ...     "PrintNode",
        ...     (),
        ...     {
        ...         "visit_match": lambda _self, _: NotImplemented,
        ...         "visit_group": lambda _self, _: NotImplemented,
        ...         "visit_expression": lambda _self, expression: print(f"Yay an "
        ...             f"expression {expression}"),
        ...         "visit_any_character": lambda _self, _: NotImplemented,
        ...         "visit_character_group": lambda _self, character_group: print(f"Yay a "
        ...             f"character group {character_group}"),
        ...         "visit_character": lambda _self, character: print(f"Yay a character {character}"),
        ...    })
        >>> printer = PrintNode()
        >>> character_matcher = Character('a')
        >>> character_matcher.accept(printer)
        Yay a character a
        >>>
        >>> character_group_matcher = CharacterGroup((CharacterRange('a', 'z'),), True)
        >>> character_group_matcher.accept(printer)
        Yay a character group [^a-z]
        >>> q = Quantifier('+', lazy=True)
        >>> expr = Expression([Match(character_matcher, q)], Expression([Match(character_group_matcher, None)]))
        >>> expr.accept(printer)
        Yay an expression Expression(seq=[Match(item=a, quantifier=Quantifier(param='+', lazy=True))], alternate=Expression(seq=[Match(item=[^a-z], quantifier=None)], alternate=None))

        """
        method_name = f"visit_{self.pattern.sub('_', self.__class__.__name__).lower()}"
        visit_method = getattr(visitor, method_name)
        return visit_method(self)

    @abstractmethod
    def to_string(self) -> str:
        """
        Converts ("reverse engineers) a regex node to its sources string
        """
        ...


class InvalidQuantifier(Exception):
    ...


@dataclass(slots=True, frozen=True)
class Quantifier:
    """
    A class representing a regular expression quantifier

    Attributes
    ----------
    param: str | tuple[int, Optional[int]]
        A generically named union of the single character quantifiers and a range quantifier
            `*`, `+`, `?` are represented by str
            Range quantifiers are represented a tuple of 2 elements, (n, m).
                See the notes section for more information
    lazy: bool
        True if the quantifier is lazy


    Raises
    ------
    InvalidQuantifier
        If an invalid parameter is given to the constructor

    Notes
    -----
    Supported are both lazy and greedy variants of:
    *	    Matches zero or more times.
    +       Matches one or more times.
    ?	    Matches zero or one time.
    {n}     Matches exactly n times. (n >= 0) ; Represented by m = None
    {n,}    Matches at least n times. (n >= 0);   Represented by m = maxsize
    {n,m}	Matches from n to m times.  (n >= 0, m >= n)
    {,m}    Matches from 0 to m times (m >= 1);  Represented by n = 0
    """

    param: str | tuple[int, Optional[int]]
    lazy: bool

    def _validate_range(self):
        """
        Validates a range quantifier

        Raises
        ------
        InvalidQuantifier
            If an invalid range is given to the constructor

        Examples
        --------
        >>> Quantifier((-1, None), lazy=False)
        Traceback (most recent call last):
            ...
        reg.parser.InvalidQuantifier: fixed quantifier: {n} must be >= 0: -1 < 0
        >>> Quantifier((-1, maxsize), lazy=False)
        Traceback (most recent call last):
            ...
        reg.parser.InvalidQuantifier: {n,} quantifier: n>=0 constraint violated: -1 < 0
        >>> Quantifier((-1, 1), lazy=False)
        Traceback (most recent call last):
            ...
        reg.parser.InvalidQuantifier: {n,m} quantifier: n>=0 constraint violated: -1 < 0
        >>> Quantifier((1, 0), lazy=False)
        Traceback (most recent call last):
            ...
        reg.parser.InvalidQuantifier: {n,m} quantifier: m>=n constraint violated: 0 < 1

        """

        assert self.param is not None
        n, m = self.param
        if m is None:
            if n < 0:
                raise InvalidQuantifier(
                    f"fixed quantifier: {{n}} must be >= 0: {n} < 0"
                )
        elif isinstance(m, int):
            if m == maxsize:
                if n < 0:
                    raise InvalidQuantifier(
                        f"{{n,}} quantifier: n>=0 constraint violated: {n} < 0"
                    )
            else:
                if n < 0:
                    raise InvalidQuantifier(
                        f"{{n,m}} quantifier: n>=0 constraint violated: {n} < 0"
                    )
                if m < n:
                    raise InvalidQuantifier(
                        f"{{n,m}} quantifier: m>=n constraint violated: {m} < {n}"
                    )
        else:
            raise InvalidQuantifier(
                f"Invalid Range Quantifier: invalid range {{{n}, {m}}}"
            )

    def _validate_single_letter_quantifier(self):
        """
        Check that a quantifier is among the supported options: ('?', '+', '*')

        Raises
        ------
        InvalidQuantifier
            If an invalid letter is given to the constructor

        Examples
        --------
        >>> Quantifier('&', lazy=False)
        Traceback (most recent call last):
            ...
        reg.parser.InvalidQuantifier: invalid quantifier '&': options are ('?', '+', '*')
        """
        if self.param not in QUANTIFIER_OPTIONS:
            raise InvalidQuantifier(
                f"invalid quantifier {self.param!r}: options are {QUANTIFIER_OPTIONS!r}"
            )

    def __post_init__(self):
        """
        Validates parameter passed to the dataclass
        See helper methods for details
        """

        if isinstance(self.param, str):
            self._validate_single_letter_quantifier()
        else:
            assert self.param is not None
            self._validate_range()

    def string(self) -> str:
        if isinstance(self.param, str):
            base = self.param
        else:
            n, m = self.param
            if m is None:
                base = f"{{{n}}}"  # {n}
            elif isinstance(m, int):
                if m == maxsize:
                    base = f"{{{n},}}"  # {n,}
                else:
                    base = f"{{{n},{m}}}"  # {n,m}
            else:
                # start == 0
                base = f"{{,{m}}}"  # {,m}
        return base + ("?" if self.lazy else "")


@dataclass
class CharacterRange(Matcher):
    start: str
    end: str

    def __call__(self, cursor: Cursor, context: Context) -> bool:
        position, text = cursor[0], context.text
        if position < len(text):
            if context.flags & RegexFlag.IGNORECASE:
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
            raise RegexpParsingError(
                f"Invalid Character Range: [{self.start}-{self.end}] is not ordered"
            )

    def __hash__(self):
        return hash((self.start, self.end))

    def __repr__(self):
        return f"{self.start}-{self.end}"


class AnchorType(Enum):
    Epsilon = "ε"
    GroupLink = "Link"
    GroupEntry = "GroupEntry"
    GroupExit = "GroupExit"

    StartOfString = "^"
    EndOfString = "$"
    EmptyString = ""

    # must be escaped
    WordBoundary = "\\b"
    NonWordBoundary = "\\B"
    StartOfStringOnly = "\\A"
    EndOfStringOnlyNotNewline = "\\z"
    EndOfStringOnlyMaybeNewLine = "\\Z"


char2anchor_type: Final[dict[str, AnchorType]] = {
    "^": AnchorType.StartOfString,
    "$": AnchorType.EndOfString,
    "b": AnchorType.WordBoundary,
    "B": AnchorType.NonWordBoundary,
    "A": AnchorType.StartOfStringOnly,
    "z": AnchorType.EndOfStringOnlyNotNewline,
    "Z": AnchorType.EndOfStringOnlyMaybeNewLine,
}


class MatchingNode(RegexNode, Matcher, ABC):
    ...


@dataclass(slots=True)
class Anchor(MatchingNode):
    anchor_type: AnchorType
    # this field is reserved for anchors which capture groups
    offset: Optional[int] = None

    @staticmethod
    def group_entry(group_index: int):
        return Anchor(AnchorType.GroupEntry, group_index * 2)

    @staticmethod
    def group_exit(group_index: int):
        return Anchor(AnchorType.GroupExit, group_index * 2 + 1)

    def __call__(self, cursor: Cursor, context: Context) -> bool:
        (text, flags), (position, _) = context, cursor
        match self.anchor_type:
            case AnchorType.StartOfString:
                # match the start of the string
                return position == 0 or (
                    # and in MULTILINE mode also matches immediately after each newline.
                    bool(RegexFlag.MULTILINE & flags)
                    and (not text or (position > 0) and text[position - 1] == "\n")
                )
            case AnchorType.EndOfString:
                # . foo matches both ‘foo’ and ‘foobar’,
                # while the regular expression foo$ matches only ‘foo’.
                # More interestingly, searching for foo.$ in `foo1\n foo2\n` matches ‘foo2’ normally,
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
                    bool(RegexFlag.MULTILINE & flags)
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
                return text and is_word_boundary(text, position)
            case AnchorType.NonWordBoundary:
                return text and not is_word_boundary(text, position)
            case AnchorType.EmptyString | AnchorType.GroupEntry | AnchorType.GroupExit:
                return True
            # By design, group links and epsilon's never match anything
            case AnchorType.GroupLink | AnchorType.Epsilon:
                return False

        raise NotImplementedError

    def _update_groups_and_index(self, cursor: Cursor) -> Cursor:
        # must create a shallow copy
        position, groups = cursor
        groups_copy = groups[:]
        groups_copy[self.offset] = position
        return Cursor(self.increment_index(position), groups_copy)

    def to_string(self):
        return self.anchor_type.value

    def __hash__(self):
        return hash(self.anchor_type)

    def __repr__(self):
        if (
            self.anchor_type == AnchorType.GroupEntry
            or self.anchor_type == AnchorType.GroupExit
        ):
            return f"{self.anchor_type.name}({self.offset >> 1})"
        return self.anchor_type.name


EPSILON: Final[Anchor] = Anchor(AnchorType.Epsilon)
GROUP_LINK: Final[Anchor] = Anchor(AnchorType.GroupLink)
EMPTY_STRING: Final[Anchor] = Anchor(AnchorType.EmptyString)


@dataclass
class AnyCharacter(MatchingNode):
    def __call__(self, cursor: Cursor, context: Context) -> bool:
        position, _ = cursor
        return position < len(context.text) and (
            bool(context.flags & RegexFlag.DOTALL) or context.text[position] != "\n"
        )

    def __repr__(self):
        return "Dot"

    def __hash__(self):
        return hash(".")

    def to_string(self):
        return "."


@dataclass
class Character(MatchingNode):
    char: str

    def __call__(self, cursor: Cursor, context: Context) -> bool:
        (text, flags), (position, _) = context, cursor
        if position < len(text):
            if flags & RegexFlag.IGNORECASE:
                return self.char.casefold() == text[position].casefold()
            return self.char == text[position]
        return False

    def __eq__(self, other) -> bool:
        return other == self.char

    def __lt__(self, other) -> bool:
        return other <= self.char

    def __repr__(self):
        return f"{self.char}"

    def __hash__(self):
        return hash(self.char)

    def to_string(self):
        if self.char in ESCAPED:
            return "\\" + self.char
        return self.char


@dataclass
class Word(MatchingNode):
    chars: str

    def to_string(self) -> str:
        return self.chars

    def __call__(self, cursor: Cursor, context: Context) -> bool:
        (text, flags), (position, _) = context, cursor
        if position < len(text):
            if flags & RegexFlag.IGNORECASE:
                return (
                    self.chars.casefold()
                    == text[position : len(self.chars) + position].casefold()
                )
            return text.startswith(self.chars, position)
        return False

    def __hash__(self) -> int:
        return hash(self.chars)

    def increment_index(self, index: int) -> int:
        return index + len(self.chars)

    def __repr__(self):
        return f"{self.chars}"


@dataclass
class CharacterGroup(MatchingNode):
    matching_nodes: tuple[Character | CharacterRange, ...]
    negated: bool = False

    def __call__(self, cursor, context) -> bool:
        if cursor.position >= len(context.text):
            return False
        return self.negated ^ any(
            matching_node(cursor, context) for matching_node in self.matching_nodes
        )

    def __eq__(self, other):
        if isinstance(other, CharacterGroup):
            return self.matching_nodes == other.matching_nodes
        return False

    def __repr__(self):
        return f"[{('^' if self.negated else '')}{''.join(map(repr, self.matching_nodes))}]"

    def __lt__(self, other):
        return id(self) < id(other)

    def __hash__(self):
        return hash((self.matching_nodes, self.negated))

    def to_string(self):
        return self.__repr__()


@dataclass
class Match(RegexNode):
    item: Character | CharacterGroup | AnyCharacter | Word
    quantifier: Optional[Quantifier]

    def to_string(self):
        return f'{self.item.to_string()}{self.quantifier.string() if self.quantifier else ""}'


@dataclass
class Group(RegexNode):
    expression: "Expression"
    index: Optional[int]
    quantifier: Optional[Quantifier]

    def is_capturing(self):
        return self.index is not None

    def to_string(self):
        expression = (
            f"({'' if self.is_capturing() else '?:'}{self.expression.to_string()})"
        )
        if self.quantifier is not None:
            return f"{expression}{self.quantifier.string()}"
        return expression


@dataclass
class Expression(RegexNode):
    seq: list[Anchor | Group | Match]
    alternate: Optional["Expression"] = None

    def to_string(self):
        seq = "".join(item.to_string() for item in self.seq)
        if self.alternate is not None:
            return f"{seq}|{self.alternate.to_string()}"
        return seq


class RegexNodesVisitor(Generic[V], metaclass=ABCMeta):
    @abstractmethod
    def visit_expression(self, expression: Expression) -> V:
        ...

    @abstractmethod
    def visit_group(self, group: Group) -> V:
        ...

    @abstractmethod
    def visit_match(self, match: Match) -> V:
        ...

    @abstractmethod
    def visit_anchor(self, anchor: Anchor) -> V:
        ...

    @abstractmethod
    def visit_any_character(self, any_character: AnyCharacter) -> V:
        ...

    @abstractmethod
    def visit_character(self, character: Character) -> V:
        ...

    @abstractmethod
    def visit_character_group(self, character_group: CharacterGroup) -> V:
        ...

    @abstractmethod
    def visit_word(self, word: Word) -> V:
        ...


class RegexParser:
    def __init__(self, regex: str, flags: RegexFlag = RegexFlag.NOFLAG):
        self._regex = regex
        self._pos = 0
        self._flags = flags
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
                    raise ValueError(f"unrecognized inline modifier {modifier}")

    def parse_regex(self) -> RegexNode:
        if self._regex == "":
            return EMPTY_STRING
        self.parse_inline_modifiers()

        if self.matches("^"):
            anchor = Anchor(char2anchor_type[self.consume_and_return()])
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
        sub_exprs = self.parse_sub_expression()
        expr = None
        if self.matches("|"):
            self.consume("|")
            expr = (
                self.parse_expression()
                if self.can_parse_sub_expression_item()
                else EMPTY_STRING
            )
        return Expression(sub_exprs, expr)

    def parse_sub_expression(self):
        # Subexpression ::= SubexpressionItem+
        sub_exprs = [self.parse_sub_expression_item()]
        while self.can_parse_sub_expression_item():
            sub_exprs.append(self.parse_sub_expression_item())
        return sub_exprs

    def parse_sub_expression_item(self):
        if self.can_parse_group():
            return self.parse_group()
        elif self.can_parse_anchor():
            return self.parse_anchor()
        else:
            return self.parse_match()

    def parse_group(self) -> Group | Expression:
        self.consume("(")
        group_index = self._group_count
        self._group_count += 1
        if self.remainder().startswith("?:"):
            self.consume("?:")
            group_index = None
            self._group_count -= 1
        if self.matches(")"):
            expression = EMPTY_STRING
        else:
            expression = self.parse_expression()
        self.consume(")")
        quantifier = None
        if self.can_parse_quantifier():
            quantifier = self.parse_quantifier()
        return Group(expression, group_index, quantifier)

    def can_parse_quantifier(self):
        return self.matches_any(("*", "+", "?", "{"))

    def parse_quantifier(self) -> Quantifier:
        if self.matches_any(("*", "+", "?")):
            param = self.consume_and_return()
        else:
            param = self.parse_range_quantifier()
        return Quantifier(param, self.optional("?"))

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
        match_item = self.parse_match_item()
        quantifier = None
        if self.can_parse_quantifier():
            quantifier = self.parse_quantifier()
        return Match(match_item, quantifier)

    def can_parse_character_group(self):
        return self.matches("[")

    def parse_character_class(self) -> CharacterGroup:
        self.consume("\\")
        if self.matches_any(("w", "W")):
            c = self.consume_and_return()
            return CharacterGroup(
                (
                    CharacterRange("0", "9"),
                    CharacterRange("A", "Z"),
                    Character("_"),
                    CharacterRange("a", "z"),
                ),
                c == "W",
            )
        elif self.matches_any(("d", "D")):
            c = self.consume_and_return()
            return CharacterGroup((CharacterRange("0", "9"),), c == "D")
        elif self.matches_any(("s", "S")):
            c = self.consume_and_return()
            return CharacterGroup(
                tuple(
                    map(
                        lambda char: Character(char),
                        [" ", "\t", "\n", "\r", "\v", "\f"],
                    )
                ),
                c == "S",
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

    def parse_character_group_item(self) -> Character | CharacterRange | CharacterGroup:
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
                return Character(self.consume_and_return())
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
                items.append(Character(self.consume_and_return()))
            self.consume("]")

        if not items:
            raise ValueError(
                f"failed parsing from {state[0]}\n"
                f"regexp = {self._regex}\n"
                f"left   = {' ' * self._pos + self._regex[self._pos:]}"
            )

        return CharacterGroup(tuple(items), negated)

    def parse_char(self):
        if self.can_parse_escaped():
            return self.parse_escaped()
        if not self.can_parse_char():
            raise ValueError(
                f"expected a char: found {self.current() if self.within_bounds() else 'EOF'}\n"
                f"regexp = {self._regex}\n"
                f"left   = {' ' * self._pos + self.remainder()}"
            )
        return Character(self.consume_and_return())

    def can_parse_escaped(self):
        return self.matches("\\") and self.matches_any(ESCAPED, 1)

    def can_parse_anchor(self):
        return (
            self.matches("\\") and self.matches_any({"A", "z", "Z", "G", "b", "B"}, 1)
        ) or self.matches_any(("^", "$"))

    def parse_escaped(self):
        self.consume("\\")
        return Character(self.consume_and_return())

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
            return AnyCharacter()
        elif self.can_parse_character_class_or_group():
            return self.parse_character_class_or_group()
        else:
            return self.parse_char()

    def parse_anchor(self):
        if self.matches("\\"):
            self.consume("\\")
            assert self.current() in ANCHORS
        return Anchor(char2anchor_type[self.consume_and_return()])

    def __repr__(self):
        return f"Parser({self._regex})"

    @property
    def flags(self):
        return self._flags


if __name__ == "__main__":
    import doctest

    doctest.testmod()
