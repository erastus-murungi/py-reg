from abc import ABC, abstractmethod
from dataclasses import dataclass
from sys import maxsize
from typing import Callable, Optional

from more_itertools import first_true, take

from src.utils import RegexFlag


@dataclass(slots=True)
class CapturingGroup:
    start: Optional[int] = None
    end: Optional[int] = None

    def copy(self):
        return CapturingGroup(self.start, self.end)

    def string(self, text: str):
        if self.start is not None and self.end is not None:
            return text[self.start : self.end]
        return None


@dataclass(frozen=True, slots=True)
class RegexMatch:
    start: int
    end: int
    text: str
    captured_groups: list[CapturingGroup]

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


@dataclass(slots=True, frozen=True)
class Cursor:
    """
    Attributes
    ----------
    text: str
            The string we are matching against this pattern
    position: int
            An index specifying the suffix of `text` (`text[index]`) against this pattern
    """

    text: str
    position: int
    flags: RegexFlag
    groups: list[CapturingGroup]


class RegexPattern(ABC):
    def __init__(self, parser):
        self.parser = parser

    @abstractmethod
    def match_suffix(self, cursor: Cursor) -> Optional[Cursor]:
        """
        Match this pattern on the substring text[index:]

        Parameters
        ----------
        cursor: Cursor
            a cursor object for the suffix

        Notes
        -----
        We only need a regex matcher to implement this method
        All the others can be constructed as long as we have this method implemented

        """

        pass

    def finditer(self, text: str):
        assert self.parser.flags is not None
        start = 0
        while start <= len(text):
            cursor = Cursor(
                text,
                start,
                self.parser.flags,
                [CapturingGroup() for _ in range(self.parser.group_count)],
            )
            if (result := self.match_suffix(cursor)) is not None:
                position, captured_groups = result.position, result.groups
                yield RegexMatch(
                    start,
                    position,
                    text,
                    captured_groups,
                )
                start = position + 1 if position == start else position
            else:
                start = start + 1

    def match(self, text: str):
        """Try to apply the pattern at the start of the string, returning
        a Match object, or None if no match was found."""
        return first_true(self.finditer(text), default=None)

    def findall(self, text):
        return [m.group(0) for m in self.finditer(text)]

    def _sub(
        self,
        string: str,
        replacer: str | Callable[[RegexMatch], str],
        count: int = maxsize,
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
        self,
        string: str,
        replacer: str | Callable[[RegexMatch], str],
        count: int = maxsize,
    ) -> tuple[str, int]:
        return self._sub(string, replacer, count)

    def sub(
        self,
        string: str,
        replacer: str | Callable[[RegexMatch], str],
        count: int = maxsize,
    ):
        return self._sub(string, replacer, count)[0]
