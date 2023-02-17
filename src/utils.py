from dataclasses import dataclass
from enum import IntFlag, auto
from typing import Generic, TypeVar

T = TypeVar("T", covariant=True)


@dataclass(frozen=True, slots=True)
class Fragment(Generic[T]):
    start: T
    end: T

    def __iter__(self):
        yield from [self.start, self.end]

    @staticmethod
    def duplicate(item: T) -> "Fragment[T]":
        return Fragment(item, item)


class RegexFlag(IntFlag):
    NOFLAG = auto()
    IGNORECASE = auto()
    MULTILINE = auto()
    DOTALL = auto()  # make dot match newline
    FREESPACING = auto()
