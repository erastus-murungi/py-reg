from pprint import pprint

from core import DFAState
from dfa import DFA
from nfa import NFA
from simplify import (
    simplify,
    simplify_character_classes,
    simplify_kleene_plus,
    simplify_lua,
)

# print(handle_lua("(ab)?cde?(abc)?d"))
# print(simplify_kleene_plus("(ab)+a(abcd)+ed"))
# nfa = NFA.from_regexp("ab*")
# minimized = DFA.from_nfa(nfa).minimize()
# print(minimized)
# NFA.from_regexp("(ab|c)+").draw_with_graphviz()


def test_simplify_character_classes_case_1():
    expanded = simplify_character_classes(r"ABC[a-x]\d")
    assert (
        expanded
        == "ABC(a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x)(0|1|2|3|4|5|6|7|8|9)"
    )


def test_simplify_lua_case_1():
    expanded = simplify_lua(r"a?")
    assert expanded == "(a|ε)"


def test_simplify_lua_case_2():
    expanded = simplify_lua(r"(ab)?")
    assert expanded == "((ab)|ε)"


def test_simplify_lua_case_3():
    expanded = simplify_lua(r"(a*)?")
    assert expanded == "((a*)|ε)"


def test_simplify_kleene_plus_case_1():
    expanded = simplify_lua(r"(ab)+a(abcd)+ed")
    assert expanded == "(ab)(ab)*a(abcd)(abcd)*ed"


def test_simply_maintains_simple_constructs():
    cases = [
        ("a", "a"),
        ("ab", "ab"),
        ("ab|cd", "ab|cd"),
        ("(ab)*", "(ab)*"),
        (".", "."),
        ("^", "^"),
        ("$", "$"),
    ]
    for test_input, expected in cases:
        assert simplify(test_input) == expected, (test_input, expected)


# nfa = NFA.from_regexp(r"((a|b*)+)?")
# nfa = NFA.from_regexp(r"(a|b)*abb(a|b)*")
# nfa = NFA.from_regexp(r"(a|b)*")
#
# nfa.draw_with_graphviz()
#
# dfa = DFA.from_nfa(nfa)
# dfa.draw_with_graphviz()
#
# dfa2 = dfa.minimize()
# dfa2.draw_with_graphviz()

# pprint(nf.epsilon_closure({nf.find_state(3), nf.find_state(13)}))
# pprint(nf.move({nf.find_state(3), nf.find_state(13)}, EPSILON))
