from itertools import groupby

from src.parser import (
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
        for belongs, members in groupby(
            expression.seq,
            key=lambda node: isinstance(node, Match)
            and node.quantifier is None
            and isinstance(node.item, (Character, Word)),
        ):
            if belongs:
                substr = ""

                for match in members:
                    if isinstance(match.item, Character):
                        substr += match.item.char
                    else:
                        substr += match.item.chars

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
