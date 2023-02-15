import re
from collections import deque
from enum import IntEnum, auto
from itertools import pairwise
from pprint import pprint
from sys import maxsize
from typing import Optional

from src.core import CapturedGroups, MatchResult, RegexPattern
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
    RegexFlag,
    RegexNodesVisitor,
    RegexParser,
)


class Type(IntEnum):
    End = auto()
    Jump = auto()
    Split = auto()
    Match = auto()
    StartCapturing = auto()
    EndCapturing = auto()
    EmptyString = auto()


class Instruction(dict):
    __slots__ = ("type", "next")

    def __init__(self, instruction_type, nxt=None, **kwargs):
        super().__init__(kwargs)
        self.type: Type = instruction_type
        self.next: Optional[Instruction] = nxt

    def __hash__(self):
        return id(self)

    def __repr__(self):
        if self.type == Type.Split:
            x, y = self["branches"]
            return f"L{id(self): <10}: Split(x=L{id(x): <10}, y=L{id(y): <10})"
        elif self.type == Type.Match:
            return f'L{id(self): <10}: Match({self["matchable"]!r})'
        elif self.type == Type.Jump:
            return f'L{id(self): <10}: Jump(to=L{id(self["to"]): <10})'
        return f'L{id(self): <10}: {self.type.name} {{{", ".join(f"{k!r}={v!r}" for k, v in self.items())}}}'


Thread = tuple[Instruction, list[CapturedGroup]]


def queue_thread(queue: deque[Thread], thread: Thread, index: int):
    def update_capturing_groups(
        group_index: int,
        captured_groups: CapturedGroups,
        *,
        is_start: bool,
    ):
        group_copy: CapturedGroup = captured_groups[group_index].copy()
        if is_start:
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

        match instruction.type:
            case Type.Jump:
                stack.append((instruction["to"], groups))

            case Type.Split:
                x, y = instruction["branches"]
                stack.append((y, groups))
                stack.append((x, groups.copy()))

            case Type.StartCapturing | Type.EndCapturing:
                stack.append(
                    (
                        instruction.next,
                        update_capturing_groups(
                            instruction["index"],
                            groups.copy(),  # make sure to pass a copy
                            is_start=instruction.type == Type.StartCapturing,
                        ),
                    )
                )
            case Type.EmptyString:
                stack.append((instruction.next, groups))

            case _:
                queue.append((instruction, groups))


class PikeVM(RegexPattern, RegexNodesVisitor[tuple[Instruction, Instruction]]):
    def __init__(self, pattern: str, flags: RegexFlag = RegexFlag.NOFLAG):
        self._parser = RegexParser(pattern, flags)
        self.start, last = self._parser.root.accept(self)
        last.next = Instruction(Type.End)

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
                if instruction.type == Type.Match:
                    matchable = instruction["matchable"]
                    if matchable.match(text, index, self._parser.flags):
                        if (next_index := matchable.increment(index)) == index:
                            # process all anchors immediately
                            queue_thread(queue, (instruction.next, groups), index)
                        else:
                            queue_thread(
                                frontier, (instruction.next, groups), next_index
                            )
                elif instruction.type == Type.End:
                    match_result = (index, groups)
                    # stop exploring threads in this queue, maybe explore higher-priority threads in `frontier`
                    break

            queue = frontier

        return match_result

    @staticmethod
    def _concat(
        instructions: list[tuple[Instruction, Instruction]]
    ) -> tuple[Instruction, Instruction]:
        for (_, a), (b, _) in pairwise(instructions):
            a.next = b
        return instructions[0][0], instructions[-1][1]

    def visit_expression(
        self, expression: Expression
    ) -> tuple[Instruction, Instruction]:
        instructions = self._concat(
            [sub_expression.accept(self) for sub_expression in expression.seq]
        )

        if expression.alternate:
            first, last = instructions
            first_alt, last_alt = expression.alternate.accept(self)
            empty = Instruction(Type.EmptyString)
            jump = Instruction(Type.Jump, first_alt, to=empty)
            split = Instruction(
                Type.Split,
                branches=(first, first_alt),
            )
            last.next = jump
            last_alt.next = empty
            return split, empty
        return instructions

    @staticmethod
    def one_or_more(instructions: tuple[Instruction, Instruction], lazy: bool):
        first, last = instructions
        empty = Instruction(Type.EmptyString)
        split = Instruction(
            Type.Split,
            empty,
            branches=((empty, first) if lazy else (first, empty)),
        )
        last.next = split
        return first, empty

    @staticmethod
    def zero_or_more(instructions: tuple[Instruction, Instruction], lazy: bool):
        empty = Instruction(Type.EmptyString)
        first, last = instructions
        split = Instruction(
            Type.Split, first, branches=(empty, first) if lazy else (first, empty)
        )
        jump = Instruction(Type.Jump, empty, to=split)
        last.next = jump
        return split, empty

    @staticmethod
    def zero_or_one(instructions: tuple[Instruction, Instruction], lazy: bool):
        empty = Instruction(Type.EmptyString)
        first, last = instructions
        split = Instruction(
            Type.Split, first, branches=(empty, first) if lazy else (first, empty)
        )
        last.next = empty
        return split, empty

    def _apply_quantifier(self, node: Group | Match) -> tuple[Instruction, Instruction]:
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

    def _apply_range_quantifier(
        self, node: Group | Match
    ) -> tuple[Instruction, Instruction]:
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
                return self.duplicate(Instruction(Type.EmptyString))

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

                empty = Instruction(Type.EmptyString)
                for _ in range(start, end):
                    first, last = self._gen_instructions_for_quantifiable(node)
                    jump = Instruction(Type.Jump, first, to=empty)
                    split = Instruction(
                        Type.Split,
                        first,
                        branches=(jump, first) if quantifier.lazy else (first, jump),
                    )
                    instructions.append((split, last))
                instructions.append(self.duplicate(empty))

        else:
            instructions = [
                self._gen_instructions_for_quantifiable(node) for _ in range(start)
            ]

        return self._concat(instructions)

    def _gen_instructions_for_quantifiable(
        self, node: Group | Match
    ) -> tuple[Instruction, Instruction]:
        """
        Helper method to generate fragments for nodes and matches
        """
        if isinstance(node, Group):
            return self._add_capturing_markers(node.expression.accept(self), node)
        return node.item.accept(self)

    @staticmethod
    def _add_capturing_markers(
        instructions: tuple[Instruction, Instruction], group: Group
    ) -> tuple[Instruction, Instruction]:
        if group.is_capturing():
            first, last = instructions
            start_capturing = Instruction(
                Type.StartCapturing, first, index=group.group_index
            )
            end_capturing = Instruction(Type.EndCapturing, index=group.group_index)
            last.next = end_capturing
            instructions = (start_capturing, end_capturing)
        return instructions

    def visit_group(self, group: Group) -> tuple[Instruction, Instruction]:
        if group.quantifier:
            return self._apply_quantifier(group)
        return self._add_capturing_markers(group.expression.accept(self), group)

    def visit_match(self, match: Match) -> tuple[Instruction, Instruction]:
        if match.quantifier:
            return self._apply_quantifier(match)
        return match.item.accept(self)

    @staticmethod
    def duplicate(item):
        return item, item

    def visit_anchor(self, anchor: Anchor) -> tuple[Instruction, Instruction]:
        if anchor is EMPTY_STRING:
            return self.duplicate(Instruction(Type.EmptyString))
        return self.duplicate(Instruction(Type.Match, matchable=anchor))

    def visit_any_character(
        self, any_character: AnyCharacter
    ) -> tuple[Instruction, Instruction]:
        return self.duplicate(Instruction(Type.Match, matchable=any_character))

    def visit_character(self, character: Character) -> tuple[Instruction, Instruction]:
        return self.duplicate(Instruction(Type.Match, matchable=character))

    def visit_character_group(
        self, character_group: CharacterGroup
    ) -> tuple[Instruction, Instruction]:
        return self.duplicate(Instruction(Type.Match, matchable=character_group))


if __name__ == "__main__":
    regex, t = (
        r"^([a-zA-Z0-9._%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6})*$",
        "erastusmurungi@gmail.com",
    )
    # regex, t = ("(ab)+?", "abab")

    pprint(list(re.finditer(regex, t)))
    pprint([m.groups() for m in re.finditer(regex, t)])

    p = PikeVM(regex)
    pprint(p.linearize())
    # pattern.graph()
    pprint(list(p.finditer(t)))

    pprint([m.groups() for m in p.finditer(t)])
