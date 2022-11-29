from typing import Final


def is_iterable(maybe_iterable):
    try:
        iter(maybe_iterable)
        return True
    except TypeError:
        return False


ALL_OPERATORS = ("|", "?", "+", "*", "^")
BIN_OPERATORS = ("^", "|")


LEFT_PAREN = "("
RIGHT_PAREN = ")"
KLEENE_CLOSURE = "*"
KLEENE_PLUS = "+"
UNION = "|"
CONCATENATION = "."
EPSILON = "ε"
LUA = "?"
CARET = "^"


PRECEDENCE: Final[dict[str, int]] = {
    LEFT_PAREN: 1,
    UNION: 2,
    CONCATENATION: 3,  # explicit concatenation operator
    LUA: 4,
    KLEENE_CLOSURE: 4,
    KLEENE_PLUS: 4,
    CARET: 5,
}


def precedence(token) -> int:
    try:
        return PRECEDENCE[token]
    except KeyError:
        return 6
