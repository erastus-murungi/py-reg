from pprint import pprint

from pyreg import DFA, NFA, handle_character_classes

# print(handle_lua("(ab)?cde?(abc)?d"))
# print(handle_kleene_sum("(ab)+a(abcd)+ed"))
# NFA("ab*").draw_with_graphviz()
# NFA("ab+").draw_with_graphviz()
# NFA("(ab|c)+").draw_with_graphviz()


def test_handling_character_classes_works_case_1():
    expanded = handle_character_classes(r"ABC[a-x]\d")
    assert (
        expanded
        == "ABC(a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x)(0|1|2|3|4|5|6|7|8|9)"
    )


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

# A = DFAState(state_id="A", is_start=True)
# B = DFAState(state_id="B")
# C = DFAState(state_id="C", is_accepting=True)
# D = DFAState(state_id="D")
# E = DFAState(state_id="E", is_accepting=True)
#
# transitions = {
#     A: {"a": B, "b": D},
#     B: {"a": C, "b": E},
#     C: {"a": B, "b": E},
#     D: {"a": C, "b": E},
#     E: {"a": E, "b": E},
# }
#
# dfa1 = DFA({A, B, C, D, E}, {"a", "b"}, transitions, A, {C, E})
# dfa1.draw_with_graphviz()
# dfa2 = dfa1.minimize()
# pprint(dfa2)
#
# s = frozenset()
#
# dfa2.draw_with_graphviz()
