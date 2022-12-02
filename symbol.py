from utils import Comparable
from typing import Generic, TypeVar
from abc import ABC, abstractmethod

T = TypeVar("T", bound=Comparable)


class Symbol(Generic[T], ABC):
    @abstractmethod
    def match(self, token: T) -> bool:
        pass


class Character(Symbol):
    def __init__(self, char: T):
        self.char = char

    def match(self, token: T) -> bool:
        return self.char == token


class CharacterInterval(Symbol):
    def __init__(self, start: T, stop: T):
        self.start = start
        self.stop = stop
        assert start <= stop

    def match(self, token: T) -> bool:
        return self.start <= token <= self.stop


class OneOfChar(Symbol):
    def __init__(self, options: set[T]):
        self.options = options

    def match(self, token: T) -> bool:
        return token in self.options


class AnyCharacter(Symbol):
    def __init__(self, ignore=(None,)):
        self.ignore = ignore

    def match(self, token: T) -> bool:
        return token not in self.ignore
