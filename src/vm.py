import re
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import pairwise
from pprint import pprint
from sys import maxsize
from typing import Hashable, Optional

from src.core import CapturedGroups, Fragment, MatchResult, RegexPattern
from src.match import CapturedGroup
from src.parser import (
    EMPTY_STRING,
    Anchor,
    AnyCharacter,
    Character,
    CharacterGroup,
    Expression,
    Group,
    Match,
    Matchable,
    RegexFlag,
    RegexNodesVisitor,
    RegexParser,
)


class Instruction(Hashable):
    def __hash__(self):
        return id(self)


@dataclass(slots=True, eq=False)
class End(Instruction):
    next: Optional[Instruction] = field(repr=False, default=None)


@dataclass(slots=True, eq=False)
class EmptyString(Instruction):
    next: Optional[Instruction] = field(repr=False, default=None)


@dataclass(slots=True, eq=False)
class Jump(Instruction):
    to: Instruction
    next: Optional["Instruction"] = field(repr=False, default=None)


@dataclass(slots=True, eq=False)
class Fork(Instruction):
    x: Instruction
    y: Instruction
    next: Optional["Instruction"] = field(repr=False, default=None)


@dataclass(slots=True, eq=False)
class Consume(Instruction):
    matchable: Matchable
    next: Optional["Instruction"] = field(repr=False, default=None)


@dataclass(slots=True, eq=False)
class Capture(Instruction):
    opening: bool
    index: int
    next: Optional["Instruction"] = field(repr=False, default=None)


Thread = tuple[Instruction, list[CapturedGroup]]


def queue_thread(queue: deque[Thread], thread: Thread, index: int):
    def update_capturing_groups(
        group_index: int,
        captured_groups: CapturedGroups,
        *,
        opening: bool,
    ):
        group_copy: CapturedGroup = captured_groups[group_index].copy()
        if opening:
            group_copy.start = index
        else:
            group_copy.end = index
        captured_groups[group_index] = group_copy
        return captured_groups

    stack: list[Thread] = [thread]

    visited = set()

    while stack:
        instruction, groups = stack.pop()

        if instruction in visited:
            continue

        visited.add(instruction)

        match instruction:
            case Jump(to):
                stack.append((to, groups))

            case Fork(x, y):
                stack.append((y, groups))
                stack.append((x, groups.copy()))

            case Capture(opening, group_index, next_instruction):
                stack.append(
                    (
                        next_instruction,
                        update_capturing_groups(
                            group_index,
                            groups.copy(),  # make sure to pass a copy
                            opening=opening,
                        ),
                    )
                )
            case EmptyString(next_instruction):
                stack.append((next_instruction, groups))

            case _:
                queue.append((instruction, groups))


class PikeVM(RegexPattern, RegexNodesVisitor[Fragment[Instruction]]):
    def __init__(self, pattern: str, flags: RegexFlag = RegexFlag.NOFLAG):
        self._parser = RegexParser(pattern, flags)
        self.start, last = self._parser.root.accept(self)
        last.next = End()

    def linearize(self) -> list[Instruction]:
        instructions = []
        current = self.start
        while current is not None:
            instructions.append(current)
            current = current.next
        return instructions

    def _match_at_index(self, text: str, start_index: int) -> MatchResult:
        captured_groups = [CapturedGroup() for _ in range(self._parser.group_count)]

        queue = deque()
        queue_thread(queue, (self.start, captured_groups), start_index)

        match_result = None

        for index in range(start_index, len(text) + 1):
            if not queue:
                break

            frontier = deque()

            while queue:
                instruction, groups = queue.popleft()
                if isinstance(instruction, Consume):
                    matchable = instruction.matchable
                    if matchable.match(text, index, self._parser.flags):
                        if (next_index := matchable.increment(index)) == index:
                            # process all anchors immediately
                            queue_thread(queue, (instruction.next, groups), index)
                        else:
                            queue_thread(
                                frontier, (instruction.next, groups), next_index
                            )
                elif isinstance(instruction, End):
                    match_result = (index, groups)
                    # stop exploring threads in this queue, maybe explore higher-priority threads in `frontier`
                    break

            queue = frontier

        return match_result

    @staticmethod
    def _concat(fragments: list[Fragment[Instruction]]) -> Fragment[Instruction]:
        for fragment1, fragment2 in pairwise(fragments):
            fragment1.end.next = fragment2.start
        return Fragment(fragments[0].start, fragments[-1].end)

    def visit_expression(self, expression: Expression) -> Fragment[Instruction]:
        instructions = self._concat(
            [sub_expression.accept(self) for sub_expression in expression.seq]
        )

        if expression.alternate:
            first, last = instructions
            first_alt, last_alt = expression.alternate.accept(self)
            empty = EmptyString()
            jump = Jump(empty, first_alt)
            split = Fork(first, first_alt, first)
            last.next = jump
            last_alt.next = empty
            return Fragment(split, empty)
        return instructions

    @staticmethod
    def one_or_more(
        instructions: Fragment[Instruction], lazy: bool
    ) -> Fragment[Instruction]:
        first, last = instructions
        empty = EmptyString()
        split = Fork(*((empty, first) if lazy else (first, empty)), empty)
        last.next = split
        return Fragment(first, empty)

    @staticmethod
    def zero_or_more(
        instructions: Fragment[Instruction], lazy: bool
    ) -> Fragment[Instruction]:
        empty = EmptyString()
        first, last = instructions
        split = Fork(*((empty, first) if lazy else (first, empty)), first)
        jump = Jump(split, empty)
        last.next = jump
        return Fragment(split, empty)

    @staticmethod
    def zero_or_one(
        instructions: Fragment[Instruction], lazy: bool
    ) -> Fragment[Instruction]:
        empty = EmptyString()
        first, last = instructions
        split = Fork(*((empty, first) if lazy else (first, empty)), first)
        last.next = empty
        return Fragment(split, empty)

    def _apply_quantifier(self, node: Group | Match) -> Fragment[Instruction]:
        quantifier = node.quantifier
        if quantifier.char is not None:
            fragment = self._gen_instructions_for_quantifiable(node)
            match quantifier.char:
                case "+":
                    return self.one_or_more(fragment, quantifier.lazy)
                case "*":
                    return self.zero_or_more(fragment, quantifier.lazy)
                case "?":
                    return self.zero_or_one(fragment, quantifier.lazy)
                case _:
                    raise RuntimeError(f"unrecognized quantifier {quantifier.char}")
        else:
            return self._apply_range_quantifier(node)

    def _apply_range_quantifier(self, node: Group | Match) -> Fragment[Instruction]:
        quantifier = node.quantifier
        start, end = quantifier.range_quantifier

        if start == 0:
            if end == maxsize:
                # 'a{0,} = a{0,maxsize}' expands to a*
                return self.zero_or_more(
                    self._gen_instructions_for_quantifiable(node), quantifier.lazy
                )
            elif end is None:
                # a{0} = ''
                return self.duplicate_instruction(EmptyString())

        if end is not None:
            if end == maxsize:
                # a{3,} expands to aaa+.
                # 'a{3,maxsize}
                instructions = [
                    self._gen_instructions_for_quantifiable(node)
                    for _ in range(start - 1)
                ] + [
                    self.one_or_more(
                        self._gen_instructions_for_quantifiable(node), quantifier.lazy
                    )
                ]
            else:
                # a{,5} = a{0,5} or a{3,5}

                instructions = [
                    self._gen_instructions_for_quantifiable(node) for _ in range(start)
                ]

                empty = EmptyString()
                for _ in range(start, end):
                    first, last = self._gen_instructions_for_quantifiable(node)
                    jump = Jump(empty, first)
                    split = Fork(
                        *((jump, first) if quantifier.lazy else (first, jump)), first
                    )
                    instructions.append(Fragment(split, last))
                instructions.append(self.duplicate_instruction(empty))

        else:
            instructions = [
                self._gen_instructions_for_quantifiable(node) for _ in range(start)
            ]

        return self._concat(instructions)

    def _gen_instructions_for_quantifiable(
        self, node: Group | Match
    ) -> Fragment[Instruction]:
        """
        Helper method to generate fragments for nodes and matches
        """
        if isinstance(node, Group):
            return self._add_capturing_markers(node.expression.accept(self), node)
        return node.item.accept(self)

    @staticmethod
    def _add_capturing_markers(
        instructions: tuple[Instruction, Instruction], group: Group
    ) -> Fragment[Instruction]:
        if group.is_capturing():
            first, last = instructions
            start_capturing = Capture(True, group.index, first)
            end_capturing = Capture(False, group.index)
            last.next = end_capturing
            instructions = Fragment(start_capturing, end_capturing)
        return instructions

    def visit_group(self, group: Group) -> Fragment[Instruction]:
        if group.quantifier:
            return self._apply_quantifier(group)
        return self._add_capturing_markers(group.expression.accept(self), group)

    def visit_match(self, match: Match) -> Fragment[Instruction]:
        if match.quantifier:
            return self._apply_quantifier(match)
        return match.item.accept(self)

    @staticmethod
    def duplicate_instruction(item: Instruction) -> Fragment[Instruction]:
        return Fragment(item, item)

    def visit_anchor(self, anchor: Anchor) -> Fragment[Instruction]:
        if anchor is EMPTY_STRING:
            return self.duplicate_instruction(EmptyString())
        return self.duplicate_instruction(Consume(anchor))

    def visit_any_character(self, any_character: AnyCharacter) -> Fragment[Instruction]:
        return self.duplicate_instruction(Consume(any_character))

    def visit_character(self, character: Character) -> Fragment[Instruction]:
        return self.duplicate_instruction(Consume(character))

    def visit_character_group(
        self, character_group: CharacterGroup
    ) -> Fragment[Instruction]:
        return self.duplicate_instruction(Consume(character_group))


if __name__ == "__main__":
    # regex, t = (
    #     r"(a*)*b",
    #     "a" * 22,
    # )
    regex, t = "((a))", "abc"

    a = time.monotonic()
    pprint(list(re.finditer(regex, t)))
    pprint([m.groups() for m in re.finditer(regex, t)])
    print(f"{time.monotonic() - a} seconds")

    a = time.monotonic()
    p = PikeVM(regex)
    pprint(list(p.finditer(t)))
    pprint([m.groups() for m in p.finditer(t)])
    print(f"{time.monotonic() - a} seconds")

    pprint(p.linearize())
    # pattern.graph()
