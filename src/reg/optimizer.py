from itertools import groupby
from typing import cast

from reg.parser import (
    Character,
    Expression,
    Group,
    Match,
    RegexNode,
    RegexNodesVisitor,
    Word,
)


class Optimizer(RegexNodesVisitor[None]):
    @staticmethod
    def run(root: RegexNode):
        root.accept(Optimizer())

    @staticmethod
    def merge_chars_to_words_in_expression(expression: Expression):
        items = []
        for can_be_merged, members in groupby(
            expression.seq,
            key=lambda node: isinstance(node, Match)  # a math node
            and node.quantifier is None  # ... with no quantifier
            and isinstance(
                node.item, (Character, Word)
            ),  # ... containing a character or a word
        ):
            if can_be_merged:
                substr = "".join(
                    match.item.char
                    if isinstance(match.item, Character)
                    else cast(Word, match.item).chars
                    for match in members
                )

                items.append(Match(Word(substr), quantifier=None))
            else:
                items.extend(members)
        expression.seq = items

    def visit_expression(self, expression: Expression) -> None:
        for subexpression in expression.seq:
            subexpression.accept(self)

        if expression.alternate is not None:
            expression.alternate.accept(self)

        self.merge_chars_to_words_in_expression(expression)
        return None

    def visit_group(self, group: Group) -> None:
        group.expression.accept(self)

    visit_anchor = (
        visit_any_character
    ) = (
        visit_character
    ) = visit_character_group = visit_word = visit_match = lambda self, _: None


if __name__ == "__main__":
    import doctest

    doctest.testmod()
