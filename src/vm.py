import re
from dataclasses import dataclass, field
from enum import IntEnum, auto
from itertools import pairwise
from pprint import pprint
from typing import Any, Optional
from collections import deque

from src.core import RegexPattern
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
    Quantifier,
    RegexFlag,
    RegexNode,
    RegexNodesVisitor,
    RegexParser,
)


class Type(IntEnum):
    End = auto()
    Jump = auto()
    Split = auto()
    MatchingPoint = auto()
    StartCapturing = auto()
    EndCapturing = auto()
    EmptyString = auto()


@dataclass(slots=True)
class Instruction:
    type: Type
    attrs: dict[str, Any] = field(default_factory=dict)
    next: Optional["Instruction"] = field(default=None, repr=False)

    def __getitem__(self, item):
        return self.attrs.__getitem__(item)


class InstructionGenerator(RegexNodesVisitor[tuple[Instruction, Instruction]]):
    def __init__(self, root: RegexNode):
        self.root = root
        self.start, last = root.accept(self)
        last.next = Instruction(Type.End)

    @staticmethod
    def _concat(instructions: list[tuple[Instruction, Instruction]]):
        for (_, a), (b, _) in pairwise(instructions):
            a.next = b
        return instructions[0][0], instructions[-1][1]

    def visit_expression(
        self, expression: Expression
    ) -> tuple[Instruction, Instruction]:
        # an expression is just a bunch of subexpressions and an alternator
        instructions = self._concat(
            [sub_expression.accept(self) for sub_expression in expression.seq]
        )

        if expression.alternate:
            first, last = instructions
            empty = Instruction(Type.EmptyString)
            jump = Instruction(Type.Jump, {"to": empty})
            alternate_instructions = expression.alternate.accept(self)
            split = Instruction(
                Type.Split,
                {"branches": (first, alternate_instructions[0])},
            )
            last.next = jump
            jump.next = alternate_instructions[0]
            alternate_instructions[1].next = empty
            return split, empty
        return instructions

    @staticmethod
    def one_or_more(instructions: tuple[Instruction, Instruction], lazy: bool):
        first, last = instructions
        empty = Instruction(Type.EmptyString)
        split = Instruction(
            Type.Split,
            {"branches": (empty, first) if lazy else (first, empty)},
            empty,
        )
        last.next = split
        return first, empty

    @staticmethod
    def zero_or_more(instructions: tuple[Instruction, Instruction], lazy: bool):
        empty = Instruction(Type.EmptyString)
        first, last = instructions
        if lazy:
            split = Instruction(Type.Split, {"branches": (empty, first)})
        else:
            split = Instruction(Type.Split, {"branches": (first, empty)})
        jump = Instruction(Type.Jump, {"to": split})
        last.next = jump
        return split, empty

    @staticmethod
    def zero_or_one(instructions: tuple[Instruction, Instruction], lazy: bool):
        empty = Instruction(Type.EmptyString)
        first, last = instructions
        if lazy:
            split = Instruction(Type.Split, {"branches": (empty, first)})
        else:
            split = Instruction(Type.Split, {"branches": (first, empty)})
        split.next = first
        last.next = empty
        return split, empty

    def _apply_quantifier(
        self, instructions: tuple[Instruction, Instruction], quantifier: Quantifier
    ) -> tuple[Instruction, Instruction]:
        match quantifier.char:
            case "+":
                return self.one_or_more(instructions, quantifier.lazy)
            case "*":
                return self.zero_or_more(instructions, quantifier.lazy)
            case "?":
                return self.zero_or_one(instructions, quantifier.lazy)
            case _:
                raise RuntimeError(f"unrecognized quantifier {quantifier.char}")

    def visit_group(self, group: Group) -> tuple[Instruction, Instruction]:
        instructions = group.expression.accept(self)
        if group.is_capturing():
            sc = Instruction(Type.StartCapturing, {"index": group.group_index})
            sc.next = instructions[0]
            ec = Instruction(Type.EndCapturing, {"index": group.group_index})
            instructions[1].next = ec
            instructions = (sc, ec)
        if group.quantifier:
            instructions = self._apply_quantifier(instructions, group.quantifier)
        return instructions

    def visit_match(self, match: Match) -> tuple[Instruction, Instruction]:
        instructions = match.item.accept(self)
        if match.quantifier:
            return self._apply_quantifier(instructions, match.quantifier)
        return instructions

    @staticmethod
    def duplicate(item):
        return item, item

    def visit_anchor(self, anchor: Anchor) -> tuple[Instruction, Instruction]:
        if anchor is EMPTY_STRING:
            return self.duplicate(Instruction(Type.EmptyString))
        return self.duplicate(Instruction(Type.MatchingPoint, {"matchable": anchor}))

    def visit_any_character(
        self, any_character: AnyCharacter
    ) -> tuple[Instruction, Instruction]:
        return self.duplicate(
            Instruction(Type.MatchingPoint, {"matchable": any_character})
        )

    def visit_character(self, character: Character) -> tuple[Instruction, Instruction]:
        return self.duplicate(Instruction(Type.MatchingPoint, {"matchable": character}))

    def visit_character_group(
        self, character_group: CharacterGroup
    ) -> tuple[Instruction, Instruction]:
        return self.duplicate(
            Instruction(Type.MatchingPoint, {"matchable": character_group})
        )


Thread = tuple[Instruction, list[CapturedGroup]]


class ThreadQueue(deque[Thread]):
    def __init__(self):
        super().__init__()

    def update_capturing_group(self, pc, groups, is_start, index):
        group_index = pc["index"]
        groups_copy = groups[:]
        captured_group_copy: CapturedGroup = groups[group_index].copy()
        if is_start:
            captured_group_copy.start = index
        else:
            captured_group_copy.end = index + 1
        groups_copy[group_index] = captured_group_copy
        self.add_thread((pc.next, groups_copy), index + 1)

    def add_thread(self, thread: Thread, index) -> bool:
        if thread in self:
            return False
        else:
            pc, groups = thread
            match pc.type:
                case Type.Jump:
                    self.add_thread((pc["to"], groups), index)
                case Type.Split:
                    x, y = pc["branches"]
                    self.add_thread((x, groups.copy()), index)
                    self.add_thread((y, groups), index)
                case Type.StartCapturing:
                    self.update_capturing_group(pc, groups, True, index)
                case Type.EndCapturing:
                    self.update_capturing_group(pc, groups, False, index)
                case Type.EmptyString:
                    self.add_thread((pc.next, groups), index)
                case _:
                    self.append(thread)


class PikeVM(RegexPattern):
    def __init__(self, pattern: str, flags: RegexFlag = RegexFlag.NOFLAG):
        self._parser = RegexParser(pattern, flags)
        self.start = InstructionGenerator(self._parser.root).start

    def _match_at_index(self, text: str, starting_index: int):
        captured_groups = [CapturedGroup() for _ in range(self._parser.group_count)]

        queue = ThreadQueue()
        queue.add_thread((self.start, captured_groups), starting_index)
        matched = None

        for index in range(starting_index, len(text) + 1):
            if not queue:
                break

            frontier = ThreadQueue()

            while queue:
                instruction, groups = queue.popleft()
                if instruction.type == Type.MatchingPoint:
                    matchable = instruction["matchable"]
                    if matchable.match(text, index, self._parser.flags):
                        if matchable.increment(index) == index:
                            # process all anchors immediately
                            queue.add_thread((instruction.next, groups), index)
                        else:
                            frontier.add_thread((instruction.next, groups), index)
                elif instruction.type == Type.End:
                    matched = (index, groups)
                    queue.clear()

            queue = frontier

        return matched


if __name__ == "__main__":
    regex, t = ("((a*)(abc|b))(c*)", "abc")

    pprint(list(re.finditer(regex, t)))
    pprint([m.groups() for m in re.finditer(regex, t)])

    p = PikeVM(regex)
    # pattern.graph()
    pprint(list(p.finditer(t)))

    pprint([m.groups() for m in PikeVM(regex).finditer(t)])
