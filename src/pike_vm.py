import re
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import pairwise
from pprint import pprint
from sys import maxsize
from typing import Final, Hashable, Optional

from src.backtracking_matcher import CapturedGroup
from src.core import CapturedGroups, Fragment, MatchResult, RegexPattern
from src.parser import (
    EMPTY_STRING,
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
    """
    Indicates that we have found a match
    Important to have the `next` attribute because each instruction has a next attribute
    """

    next: Final = None


@dataclass(slots=True, eq=False)
class EmptyString(Instruction):
    next: Optional[Instruction] = field(repr=False, default=None)


@dataclass(slots=True, eq=False)
class Jump(Instruction):
    """
    An unconditional jump to another instruction

    Attributes
    ----------
    target: Instruction
        The target instruction of this unconditional jump
    next: Instruction
        The instruction to follow when after printing this jump
        It is not actually followed during the matching process
    """

    target: Instruction
    next: Instruction


@dataclass(slots=True, eq=False)
class Fork(Instruction):
    """
    Indicates that we can follow two paths in the NFA.
    The number of paths is a max of 2 at all times.
    We usually create forks because of instructions like '+', '-', '|', '?'


    Attributes
    ----------
    preferred: Instruction
        The first branch of the split instruction that we explore
    alternative: Instruction
        The alternative branch we explore after exploring preferred without success
    next: Instruction
        The instruction to follow when after printing this fork

    Notes
    -----
    The next attribute here is only used when printing the instruction list in sequence
    It is never actually followed during the matching process because a fork doesn't consume any characters

    """

    preferred: Instruction
    alternative: Instruction
    next: Instruction


@dataclass(slots=True, eq=False)
class Consume(Instruction):
    matchable: Matchable
    next: Optional["Instruction"] = field(repr=False, default=None)


@dataclass(slots=True, eq=False)
class Capture(Instruction):
    index: int
    opening: bool
    next: Optional["Instruction"] = field(repr=False, default=None)


Thread = tuple[Instruction, list[CapturedGroup]]


class RegexPikeVM(RegexPattern, RegexNodesVisitor[Fragment[Instruction]]):
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

    @staticmethod
    def update_capturing_groups(
        group_index: int,
        captured_groups: CapturedGroups,
        text_index: int,
        *,
        opening: bool,
    ) -> CapturedGroups:
        """
        Update the capturing group at index `group_index` so that it records `text_index`

        Always makes a shallow copy of captured groups

        Parameters
        ----------
        group_index: int
            The group index of the capturing group to be updated
        captured_groups: CapturedGroups
            The collection of CapturedGroup items.
        text_index: int
            The index of the text where some group is captured
        opening: bool
            True if the captured part is the opening part of a group `(` or the closing part of a group `)`

        Returns
        -------
        A shallow copy of the passed in captured groups list with the item captured_groups[group_index] updated.

        """

        # make a shallow copy of the list of captured groups
        captured_groups_copy = captured_groups.copy()
        # make a deep copy of the actual captured group we are modifying
        group_copy: CapturedGroup = captured_groups_copy[group_index].copy()
        if opening:
            group_copy.start = text_index
        else:
            group_copy.end = text_index
        captured_groups_copy[group_index] = group_copy
        return captured_groups_copy

    @staticmethod
    def queue_thread(
        queue: deque[Thread], thread: Thread, text_index: int, visited: set[Instruction]
    ) -> None:
        """
        Queue a thread

        Parameters
        ----------
        queue: deque[Thread]
            A queue to add the thread to
        thread: Thread
            The thread to enqueue
        text_index: int
            The index we are at in the text. This is need to update capturing groups correctly
        visited: set[Instruction]
            A set of instructions which has already been visited in this lockstep
            We maintain a separate visited set during each lockstep to prevent duplicates in the `queue`
            Two threads with the same PC will execute identically even if they have different captured groups;
            thus only one thread per PC needs to be kept.

        Returns
        -------
        None:
            To indicate the function ran to completion

        Notes
        -----
        All the threads inside the queue are at the exact same text index

        We consider all alternatives of a fork step simultaneously,
        in lockstep with respect to the current position in the input string

        """

        # we use an explicit stack to traverse the instructions instead of recursion
        stack: list[Thread] = [thread]

        while stack:
            instruction, groups = stack.pop()

            if instruction in visited:
                continue

            visited.add(instruction)

            match instruction:
                case Jump(to):
                    stack.append((to, groups))

                case Fork(preferred, alternative):
                    stack.append((alternative, groups))
                    stack.append((preferred, groups.copy()))

                case Capture(group_index, opening, next_instruction):
                    stack.append(
                        (
                            next_instruction,
                            RegexPikeVM.update_capturing_groups(
                                group_index,
                                groups,  # make sure to pass a copy
                                text_index,
                                opening=opening,
                            ),
                        )
                    )
                case EmptyString(next_instruction):
                    stack.append((next_instruction, groups))

                case _:
                    queue.append((instruction, groups))

    def _match_at_index(self, text: str, start_index: int) -> MatchResult:
        captured_groups = [CapturedGroup() for _ in range(self._parser.group_count)]

        queue, visited = deque(), set()
        self.queue_thread(queue, (self.start, captured_groups), start_index, visited)

        match_result = None

        for index in range(start_index, len(text) + 1):
            if not queue:
                break

            frontier, next_visited = deque(), set()

            while queue:
                instruction, groups = queue.popleft()
                if isinstance(instruction, Consume):
                    matchable = instruction.matchable
                    if matchable.match(text, index, self._parser.flags):
                        if (next_index := matchable.increment(index)) == index:
                            # process all anchors immediately
                            self.queue_thread(
                                queue, (instruction.next, groups), index, visited
                            )
                        else:
                            self.queue_thread(
                                frontier,
                                (instruction.next, groups),
                                next_index,
                                next_visited,
                            )
                elif isinstance(instruction, End):
                    match_result = (index, groups)
                    # stop exploring threads in this queue, maybe explore higher-priority threads in `frontier`
                    break

            queue, visited = frontier, next_visited

        return match_result

    @staticmethod
    def _concat_fragments(
        fragments: list[Fragment[Instruction]],
    ) -> Fragment[Instruction]:
        for fragment1, fragment2 in pairwise(fragments):
            fragment1.end.next = fragment2.start
        return Fragment(fragments[0].start, fragments[-1].end)

    def visit_expression(self, expression: Expression) -> Fragment[Instruction]:
        """
        Parameters
        ----------
        expression: Expression
            The expression to convert to a sequence of instructions

        Notes
        ----

            Alternate e1|e2       fork L1, L2
            (first, last)         L1: codes for e1
                                  jump L3
            (first_alt, last_alt) L2: codes for e2
            (empty)               L3:
        """
        codes = self._concat_fragments(
            [sub_expression.accept(self) for sub_expression in expression.seq]
        )

        if expression.alternate:
            alt_codes = expression.alternate.accept(self)
            empty = EmptyString()
            codes.end.next = Jump(empty, alt_codes.start)
            alt_codes.end.next = empty
            return Fragment(Fork(codes.start, alt_codes.start, codes.start), empty)
        return codes

    @staticmethod
    def one_or_more(codes: Fragment[Instruction], lazy: bool) -> Fragment[Instruction]:
        """
        Notes
        -----
        L1: codes for e
            fork L1, L3
        L3: empty
        """
        empty = EmptyString()
        codes.end.next = Fork(
            *((empty, codes.start) if lazy else (codes.start, empty)), empty
        )
        return Fragment(codes.start, empty)

    @staticmethod
    def zero_or_more(codes: Fragment[Instruction], lazy: bool) -> Fragment[Instruction]:
        """
        Notes
        -----
        L1: fork L2, L3
        L2: codes for e
            jmp L1
        L3:
        """
        empty = EmptyString()
        split = Fork(
            *((empty, codes.start) if lazy else (codes.start, empty)), codes.start
        )
        codes.end.next = Jump(split, empty)
        return Fragment(split, empty)

    @staticmethod
    def zero_or_one(codes: Fragment[Instruction], lazy: bool) -> Fragment[Instruction]:
        """
        Notes
        -----
        fork L1, L2
        L1: codes for e
        L2:
        """
        empty = EmptyString()
        codes.end.next = empty
        return Fragment(
            Fork(
                *((empty, codes.start) if lazy else (codes.start, empty)), codes.start
            ),
            empty,
        )

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
                return Fragment.duplicate(EmptyString())

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

                empty: Final[EmptyString] = EmptyString()
                for _ in range(start, end):
                    first, last = self._gen_instructions_for_quantifiable(node)
                    jump = Jump(empty, first)
                    split = Fork(
                        *((jump, first) if quantifier.lazy else (first, jump)), first
                    )
                    instructions.append(Fragment(split, last))
                instructions.append(Fragment.duplicate(empty))

        else:
            instructions = [
                self._gen_instructions_for_quantifiable(node) for _ in range(start)
            ]

        return self._concat_fragments(instructions)

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
        codes: Fragment[Instruction], group: Group
    ) -> Fragment[Instruction]:
        if group.is_capturing():
            # All we are doing here is just wrapping our fragment between two capturing instructions

            start_capturing, end_capturing = Capture(
                group.index, opening=True
            ), Capture(group.index, opening=False)

            start_capturing.next = codes.start
            codes.end.next = end_capturing
            return Fragment(start_capturing, end_capturing)

        return codes

    def visit_quantifiable(self, node: Group | Match) -> Fragment[Instruction]:
        if node.quantifier:
            return self._apply_quantifier(node)
        return self._gen_instructions_for_quantifiable(node)

    visit_group = visit_match = visit_quantifiable

    visit_character = (
        visit_character_group
    ) = visit_any_character = visit_anchor = lambda _, matchable: Fragment.duplicate(
        EmptyString() if matchable is EMPTY_STRING else Consume(matchable)
    )


if __name__ == "__main__":
    # regex, t = (
    #     r"(a*)*b",
    #     "a" * 22,
    # )
    regex, t = ("($)|()", "xxx")
    a = time.monotonic()
    pprint(list(re.finditer(regex, t)))
    pprint([m.groups() for m in re.finditer(regex, t)])
    print(f"{time.monotonic() - a} seconds")

    a = time.monotonic()
    p = RegexPikeVM(regex)
    pprint(list(p.finditer(t)))
    pprint([m.groups() for m in p.finditer(t)])
    print(f"{time.monotonic() - a} seconds")

    # pprint(p.linearize())
