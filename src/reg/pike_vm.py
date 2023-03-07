from collections import deque
from dataclasses import dataclass, field
from itertools import pairwise
from sys import maxsize
from typing import Final, Hashable, Optional

from reg.matcher import Context, Cursor, RegexPattern
from reg.optimizer import Optimizer
from reg.parser import (
    EMPTY_STRING,
    Anchor,
    Expression,
    Group,
    Match,
    Matcher,
    RegexFlag,
    RegexNodesVisitor,
    RegexParser,
)
from reg.utils import Fragment


class Instruction(Hashable):
    __slots__ = "next"

    next: Optional["Instruction"]

    def __hash__(self):
        return id(self)


@dataclass(slots=True, eq=False, frozen=True)
class End(Instruction):
    """
    Indicates that we have found a match
    Important to have the `next` attribute because each instruction has a next attribute
    """

    next = None


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
    matcher: Matcher
    next: Optional["Instruction"] = field(repr=False, default=None)


@dataclass(slots=True, eq=False)
class Capture(Instruction):
    capturing_anchor: Anchor
    next: Optional["Instruction"] = field(repr=False, default=None)


Thread = tuple[Optional[Instruction], Cursor]
VMSearchSpaceNode = tuple[int, Optional[Instruction]]


class RegexPikeVM(RegexPattern, RegexNodesVisitor[Fragment[Instruction]]):
    """
    Examples
    --------
    >>> from sys import maxsize
    >>> pattern, text = '(ab)+', 'abab'
    >>> compiled_regex = RegexPikeVM(pattern)
    >>> ctx = Context(text, RegexFlag.NOFLAG)
    >>> start = 0
    >>> c = compiled_regex.match_suffix(Cursor(start, (maxsize, maxsize)), ctx)
    >>> c
    Cursor(position=4, groups=(2, 4))
    >>> end, groups = c
    >>> assert text[start: end] == 'abab'
    """

    def __init__(self, pattern: str, flags: RegexFlag = RegexFlag.OPTIMIZE):
        parser = RegexParser(pattern, flags)
        super().__init__(parser.group_count, parser.flags)
        if parser.should_optimize():
            parser.root.accept(Optimizer())
        self.start, last = parser.root.accept(self)
        last.next = End()

    def linearize(self) -> list[Instruction]:
        instructions = []
        current = self.start
        while current is not None:
            instructions.append(current)
            current = current.next
        return instructions

    @staticmethod
    def queue_thread(
        queue: deque[Thread], thread: Thread, visited: set[VMSearchSpaceNode]
    ) -> None:
        """
        Queue a thread

        Parameters
        ----------
        queue: deque[Thread]
            A queue to add the thread to
        thread: Thread
            The thread to enqueue
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
            instruction, cursor = stack.pop()

            if (cursor.position, instruction) in visited:
                continue

            visited.add((cursor.position, instruction))

            match instruction:
                case Jump(target):
                    stack.append((target, cursor))

                case Fork(preferred, alternative):
                    stack.extend(((alternative, cursor), (preferred, cursor)))

                case Capture(capturing_anchor, next_instruction):
                    stack.append((next_instruction, capturing_anchor.update(cursor)))

                case EmptyString(next_instruction):
                    stack.append((next_instruction, cursor))

                case _:
                    queue.append((instruction, cursor))

    def match_suffix(self, cursor: Cursor, context: Context) -> Optional[Cursor]:
        queue: deque[Thread] = deque()

        visited: set[VMSearchSpaceNode] = set()
        self.queue_thread(queue, (self.start, cursor), set())

        match = None

        while True:
            frontier, next_visited = (
                deque(),
                set(),
            )  # type: (deque[Thread], set[VMSearchSpaceNode])

            while queue:
                instruction, cursor = queue.popleft()

                match instruction:
                    case Consume(matcher, next_instruction):
                        if matcher(cursor, context):
                            next_cursor = matcher.update_index(cursor)
                            if next_cursor.position == cursor.position:
                                # process all anchors immediately
                                self.queue_thread(
                                    queue, (next_instruction, next_cursor), visited
                                )
                            else:
                                self.queue_thread(
                                    frontier,
                                    (next_instruction, next_cursor),
                                    next_visited,
                                )
                    case End():
                        match = cursor
                        # stop exploring threads in this queue, maybe explore higher-priority threads in `frontier`
                        break
            if not frontier:
                break
            queue, visited = frontier, next_visited

        return match

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

    def _apply_range_quantifier(
        self, node: Group | Match, n: int, m: Optional[int], lazy: bool
    ) -> Fragment[Instruction]:
        if n == 0:
            if m == maxsize:
                # 'a{0,} = a{0,maxsize}' expands to a*
                return self.zero_or_more(
                    self._gen_instructions_for_quantifiable(node), lazy
                )
            elif m is None:
                # a{0} = ''
                return Fragment.duplicate(EmptyString())

        if m is not None:
            if m == maxsize:
                # a{3,} expands to aaa+.
                # 'a{3,maxsize}
                instructions = [
                    self._gen_instructions_for_quantifiable(node) for _ in range(n - 1)
                ] + [
                    self.one_or_more(
                        self._gen_instructions_for_quantifiable(node), lazy
                    )
                ]
            else:
                # a{,5} = a{0,5} or a{3,5}

                instructions = [
                    self._gen_instructions_for_quantifiable(node) for _ in range(n)
                ]

                empty: Final[EmptyString] = EmptyString()
                for _ in range(n, m):
                    first, last = self._gen_instructions_for_quantifiable(node)
                    jump = Jump(empty, first)
                    split = Fork(*((jump, first) if lazy else (first, jump)), first)
                    instructions.append(Fragment(split, last))
                instructions.append(Fragment.duplicate(empty))

        else:
            instructions = [
                self._gen_instructions_for_quantifiable(node) for _ in range(n)
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
        if group.index is not None:
            # All we are doing here is just wrapping our fragment between two capturing instructions
            start_capturing, end_capturing = Capture(
                Anchor.group_entry(group.index)
            ), Capture(Anchor.group_exit(group.index))

            start_capturing.next = codes.start
            codes.end.next = end_capturing
            return Fragment(start_capturing, end_capturing)

        return codes

    def visit_quantifiable(self, node: Group | Match) -> Fragment[Instruction]:
        quantifier = node.quantifier
        if quantifier is not None:
            if isinstance(quantifier.param, str):
                fragment = self._gen_instructions_for_quantifiable(node)
                match quantifier.param:
                    case "+":
                        return self.one_or_more(fragment, quantifier.lazy)
                    case "*":
                        return self.zero_or_more(fragment, quantifier.lazy)
                    case "?":
                        return self.zero_or_one(fragment, quantifier.lazy)
                    case _:
                        raise RuntimeError(
                            f"unrecognized quantifier {quantifier.param}"
                        )
            else:
                start, end = quantifier.param
                return self._apply_range_quantifier(node, start, end, quantifier.lazy)
        return self._gen_instructions_for_quantifiable(node)

    visit_group = visit_match = visit_quantifiable  # type: ignore

    visit_character = (  # type: ignore
        visit_character_group
    ) = (
        visit_any_character
    ) = visit_anchor = visit_word = lambda _, matcher: Fragment.duplicate(
        EmptyString() if matcher is EMPTY_STRING else Consume(matcher)  # type: ignore
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
