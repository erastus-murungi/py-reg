from abc import ABC, abstractmethod
from dataclasses import dataclass
from sys import maxsize
from typing import Callable, Generic, Optional, TypeVar

from more_itertools import first_true, take

from .parser import Match

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Fragment(Generic[T]):
    start: T
    end: T

    def __iter__(self):
        yield from [self.start, self.end]

    @staticmethod
    def duplicate(item: T) -> "Fragment[T]":
        return Fragment(item, item)


@dataclass(slots=True)
class CapturedGroup:
    start: Optional[int] = None
    end: Optional[int] = None

    def copy(self):
        return CapturedGroup(self.start, self.end)

    def string(self, text: str):
        if self.start is not None and self.end is not None:
            return text[self.start : self.end]
        return None


CapturedGroups = list[CapturedGroup]
MatchResult = tuple[int, CapturedGroups]


@dataclass(frozen=True, slots=True)
class Match:
    start: int
    end: int
    text: str
    captured_groups: CapturedGroups

    @property
    def span(self):
        return self.start, self.end

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(span={self.span}, "
            f"match={self.text[self.start:self.end]!r})"
        )

    def groups(self) -> tuple[str, ...]:
        return tuple(
            captured_group.string(self.text) for captured_group in self.captured_groups
        )

    def group(self, index: int = 0) -> Optional[str]:
        if index < 0 or index > len(self.captured_groups):
            raise IndexError(f"index should be 0 <= {len(self.captured_groups)}")
        if index == 0:
            return self.text[self.start : self.end]
        return self.captured_groups[index - 1].string(self.text)


class RegexPattern(ABC):
    @abstractmethod
    def _match_at_index(self, text: str, index: int) -> Optional[MatchResult]:
        ...

    def finditer(self, text: str):
        index = 0
        while index <= len(text):
            if (result := self._match_at_index(text, index)) is not None:
                position, captured_groups = result
                yield Match(
                    index,
                    position,
                    text,
                    captured_groups,
                )
                index = position + 1 if position == index else position
            else:
                index = index + 1

    def match(self, text: str):
        """Try to apply the pattern at the start of the string, returning
        a Match object, or None if no match was found."""
        return first_true(self.finditer(text), default=None)

    def findall(self, text):
        return [m.group(0) for m in self.finditer(text)]

    def _sub(
        self, string: str, replacer: str | Callable[[Match], str], count: int = maxsize
    ) -> tuple[str, int]:
        if isinstance(replacer, str):

            def r(_):
                return replacer

        else:
            r = replacer
        matches = take(count, self.finditer(string))
        chunks = []
        start = 0
        subs = 0
        for match in matches:
            chunks.append(string[start : match.start])
            chunks.append(r(match))
            start = match.end
            subs += 1
        chunks.append(string[start:])
        return "".join(chunks), subs

    def subn(
        self, string: str, replacer: str | Callable[[Match], str], count: int = maxsize
    ) -> tuple[str, int]:
        return self._sub(string, replacer, count)

    def sub(
        self, string: str, replacer: str | Callable[[Match], str], count: int = maxsize
    ):
        return self._sub(string, replacer, count)[0]
