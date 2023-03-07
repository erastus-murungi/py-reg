from enum import IntFlag, auto
from typing import Generic, NamedTuple, TypeVar

T = TypeVar("T")


class Fragment(NamedTuple, Generic[T]):
    start: T
    end: T

    @staticmethod
    def duplicate(item: T) -> "Fragment[T]":
        return Fragment(item, item)


class RegexFlag(IntFlag):
    NO_BACKTRACK = auto()
    NOFLAG = auto()
    IGNORECASE = auto()
    MULTILINE = auto()
    DOTALL = auto()  # make dot match newline
    FREESPACING = auto()
    OPTIMIZE = auto()
    DEBUG = auto()
