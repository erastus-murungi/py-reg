from collections import defaultdict
from dataclasses import dataclass, field
from functools import reduce
from itertools import chain, combinations, count, product
from operator import add
from string import ascii_lowercase, ascii_uppercase, digits
from typing import Callable, ClassVar, Final, Optional, Iterable

import graphviz

LEFT_PAREN = "("
RIGHT_PAREN = ")"
KLEENE_CLOSURE = "*"
KLEENE_PLUS = "+"
UNION = "|"
CONCATENATION = "."
EPSILON = "ε"

PRECEDENCE: Final[dict[str, int]] = {
    "(": 1,
    "|": 2,
    ".": 3,  # explicit concatenation operator
    "?": 4,
    "*": 4,
    "+": 4,
    "^": 5,
}


def precedence(token) -> int:
    try:
        return PRECEDENCE[token]
    except KeyError:
        return 6


ALL_OPERATORS = ("|", "?", "+", "*", "^")
BIN_OPERATORS = ("^", "|")


@dataclass(kw_only=True)
class State:
    next_state_id: ClassVar = count(0)
    is_start: bool = field(default_factory=lambda: False, repr=False)
    is_accepting: bool = field(default_factory=lambda: False, repr=False)
    state_id: int | str = field(default_factory=lambda: next(State.next_state_id))

    def __repr__(self):
        return str(self.state_id)

    def __hash__(self):
        return hash(self.state_id)

    def __lt__(self, other):
        return self.state_id < other.state_id


EMPTY_STATE = State(is_start=False, is_accepting=False, state_id=-1)

Symbol = str


def yield_letters():
    it = map(
        lambda t: "".join(t),
        chain.from_iterable((product(ascii_uppercase, repeat=i) for i in range(10))),
    )
    _ = next(it)
    yield from it


@dataclass(kw_only=True)
class DFAState(State):
    from_states: frozenset[State] = frozenset({})
    labels_cache: ClassVar[dict[frozenset[State], Symbol]] = {}
    labels_gen: ClassVar = yield_letters()
    max_label: ClassVar = "A"

    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        return self.state_id

    @classmethod
    def get_label(cls, frozen):
        if frozen in cls.labels_cache:
            return cls.labels_cache[frozen]
        else:
            label = next(cls.labels_gen)
            while label <= cls.max_label:
                label = next(cls.labels_gen)
            cls.labels_cache[frozen] = label
            return label

    def __post_init__(self):
        if isinstance(self.state_id, int):
            self.state_id = self.get_label(self.from_states)
        self.__class__.max_label = max(self.state_id, self.__class__.max_label)


def format_regexp(regexp: str) -> str:
    if len(regexp) == 0:
        return EPSILON
    fmt_regexp = []
    for c1, c2 in zip(regexp, regexp[1:]):
        fmt_regexp.append(c1)
        if (
            (c1 != "(" and c2 != ")")
            and c2 not in ALL_OPERATORS
            and c1 not in BIN_OPERATORS
        ):
            fmt_regexp.append(".")
    fmt_regexp.append(regexp[-1])
    return "".join(fmt_regexp)


def shunting_yard(infix: str) -> str:
    stack = []
    postfix = []
    fmt_infix = format_regexp(infix)
    for c in fmt_infix:
        if c == "(":
            stack.append(c)
        elif c == ")":
            while stack[-1] != "(":
                postfix.append(stack.pop())
            stack.pop()
        else:
            while stack:
                peeked = stack[-1]
                peeked_precedence = precedence(peeked)
                current_precedence = precedence(c)
                if peeked_precedence >= current_precedence:
                    postfix.append(stack.pop())
                else:
                    break
            stack.append(c)
    while stack:
        postfix.append(stack.pop())
    return "".join(postfix)


def handle_kleene_sum(regexp: str):
    """A+ -> AA*"""
    len_regexp = len(regexp)
    i = 0
    while i < len_regexp:
        if regexp[i] == "+":
            # two cases can arise:
            # 1. + is after a symbol, in which case + affects only that symbol
            # 2. + is after a ')', in which case + affects a whole expression inside '(' and ')'

            if (symbol := regexp[i - 1]) != ")":
                assert regexp[i - 1] not in ALL_OPERATORS and regexp[i - 1] not in (
                    "(",
                    ")",
                )
                sub_exp = symbol + symbol + "*"
                regexp = regexp[: i - 1] + sub_exp + regexp[i + 1 :]
                i += 2  # we added two extra characters
            else:
                # first find lower bound
                bracket_counter = 0
                for j in reversed(range(i)):
                    if j != i - 1 and regexp[j] == ")":
                        bracket_counter += 1
                    # stop
                    if regexp[j] == "(":
                        if bracket_counter != 0:
                            bracket_counter -= 1
                        else:
                            symbol_sequence_with_brackets = regexp[j:i]
                            sub_exp = (
                                symbol_sequence_with_brackets
                                + symbol_sequence_with_brackets
                                + "*"
                            )

                            left = regexp[:j]
                            right = regexp[i + 1 :]
                            regexp = left + sub_exp + right
                            i += len(symbol_sequence_with_brackets)
            len_regexp = len(regexp)
        i += 1
    return regexp


def handle_lua(regexp: str) -> str:
    """A? represents A|ε"""
    len_regexp = len(regexp)
    i = 0
    while i < len_regexp:
        if regexp[i] == "?":
            # two cases can arise:
            # 1. ? is after a symbol, in which case ? affects only that symbol
            # 2. ? is after a ')', in which case ? affects a whole expression inside '(' and ')'
            if (symbol := regexp[i - 1]) != ")":
                sub_exp = "(" + symbol + "|ε)"
                regexp = regexp[: i - 1] + sub_exp + regexp[i + 1 :]
                i += 3  # we replaced 1 character with 4 so we advance i by 3
            else:
                assert regexp[i - 1] == ")"
                for j in reversed(range(i)):
                    if regexp[j] == "(":
                        symbol_sequence_with_brackets = regexp[j:i]
                        sub_exp = "(" + symbol_sequence_with_brackets + "|ε)"
                        regexp = regexp[:j] + sub_exp + regexp[i + 1 :]
                        i += 3
                        break
            len_regexp = len(regexp)
        i += 1
    return regexp


def handle_alpha(char_start, chart_end):
    if char_start.isalpha() and chart_end.isalpha():
        letters_sorted = ascii_uppercase + ascii_lowercase
        if char_start > chart_end:
            raise ValueError(f"illegal character range: [{char_start}-{chart_end}]")
        else:
            return letters_sorted[
                letters_sorted.index(char_start) : letters_sorted.index(chart_end) + 1
            ]


def handle_digits_case(char_start, char_end):
    if char_start.isdigit() and char_end.isdigit():
        if char_start > char_end:
            raise ValueError(f"illegal character range: [{char_start}-{char_end}]")
        else:
            return digits[digits.index(char_start) : digits.index(char_end) + 1]


def handle_range(char_start, char_end):
    if not char_start.isalnum() or not char_end.isalnum():
        raise ValueError(f"Illegal character range: [{char_start}-{char_end}]")
    # same letter
    if char_start == char_end:
        return char_start

    return handle_alpha(char_start, char_end) or handle_digits_case(
        char_start, char_end
    )


def parse_character_class(regexp: str):
    if len(regexp) < 2:
        raise ValueError(f"invalid character class{regexp}")
    assert regexp[0] == "[" and regexp[-1] == "]"
    assert all(
        special_char not in regexp for special_char in ALL_OPERATORS + ("(", ")")
    )
    if "-" in regexp:
        regexp = handle_ranges(regexp)
    else:
        regexp = regexp[1:-1]
    return LEFT_PAREN + UNION.join(regexp) + RIGHT_PAREN


def handle_ranges(regexp: str):
    assert regexp[0] == "[" and regexp[-1] == "]"
    assert regexp.count("[") == 1 and regexp.count("]") == 1
    assert len(regexp) > 1

    len_regexp = len(regexp)
    i = 1
    subs = []
    while i < len_regexp - 1:
        if regexp[i] == "-":
            sub_regexp = handle_range(regexp[i - 1], regexp[i + 1])
            subs.pop()
            subs.append(sub_regexp)
            i += 2
        else:
            subs.append(regexp[i])
            i += 1
    return "".join(subs)


def simplify_character_classes(regexp: str):
    return (
        regexp.replace("\\d", "[0-9]")
        .replace("\\w", "[A-z0-9_]")
        .replace("\\s", "[ \t\n\r\v\f]")
    )


def handle_character_classes(regexp: str):
    regexp = simplify_character_classes(regexp)
    subs = []
    s = None
    for i in range(len(regexp)):
        if regexp[i] == "]":
            if s is not None:
                sub_regexp = parse_character_class(regexp[s : i + 1])
                subs.append(sub_regexp)
                s = None
            else:
                raise ValueError(
                    f"unescaped closing bracket found at regexp[{i}] before any opening bracket"
                )
        elif regexp[i] == "[":
            if s is not None:
                raise ValueError(
                    f"found another opening bracket before the one at {s} was closed"
                )
            s = i
        elif s is None:
            subs.append(regexp[i])
    if s is not None:
        raise ValueError(
            f"could not find closing square bracket to the one opened at regexp[{s}]"
        )
    return "".join(subs)


def simplify_extensions(regexp: str) -> str:
    return handle_lua(handle_kleene_sum(handle_character_classes(regexp)))


class NFA:
    """Formally, an NFA is a 5-tuple (Q, Σ, q0, T, δ) where
        • Q is finite set of states;
        • Σ is alphabet of input symbols;
        • q0 is start state;
        • T is subset of Q giving the accept states;
        and
        • δ is the transition function.
    Now the transition function specifies a set of states rather than a state: it maps Q × Σ to { subsets of Q }."""

    save_final: Optional[State]

    def __init__(
        self,
        states: set[State],
        symbols: set[Symbol],
        transition_table: dict[State, dict[Symbol, Iterable]]
        | Callable[[State, Symbol], State]
        | dict[State, dict[Symbol, State]]
        | Callable[[State, Symbol], Iterable],
        start_state: State,
        accepting_states: set[State] | frozenset[State],
    ):
        self.states = states
        self.symbols = symbols
        self.transition_table = transition_table
        self.start_state = start_state
        self.accept_states = accepting_states

    @staticmethod
    def from_regexp(regexp: str) -> "NFA":
        postfix_regexp: str = shunting_yard(simplify_extensions(regexp))
        states, symbols, start_state, final_state, transition_table = NFA.regexp_to_nfa(
            postfix_regexp
        )
        return NFA(states, symbols, transition_table, start_state, {final_state})

    @staticmethod
    def get_transition_function(transition_table) -> Callable[[State, Symbol], State]:
        def _transition_function(state, symbol):
            assert isinstance(state, State)
            assert isinstance(symbol, Symbol)
            try:
                return transition_table[state][symbol]
            except KeyError:
                return EMPTY_STATE

        return _transition_function

    @property
    def transitions(self):
        for s1, table in self.transition_table.items():
            for symbol, s2s in table.items():
                if is_iterable(s2s):
                    for s2 in s2s:
                        yield (symbol, s1, s2)
                else:
                    yield (symbol, s1, s2s)

    def make_transition(self, state, symbol):
        try:
            return self.transition_table[state][symbol]
        except KeyError:
            return EMPTY_STATE

    def states_eq(self, state1: State, state2: State):
        # both states should be accepting or both non_accepting
        if state1.is_accepting ^ state2.is_accepting:
            return False
        for symbol in self.symbols:
            if self.make_transition(state1, symbol) != self.make_transition(
                state2, symbol
            ):
                return False
        return True

    @staticmethod
    def get_states_pair():
        return State(), State()

    # noinspection DuplicatedCode
    @staticmethod
    def union(
        lower_start: State,
        upper_start: State,
        lower_accept: State,
        upper_accept: State,
        start_states_stack: list[State],
        accept_states_stack: list[State],
        transition_table: dict[State, dict[Symbol, list[State]]],
    ) -> None:
        new_start, new_accept = NFA.get_states_pair()

        transition_table[new_start][EPSILON].append(lower_start)
        transition_table[new_start][EPSILON].append(upper_start)
        transition_table[lower_accept][EPSILON].append(new_accept)
        transition_table[upper_accept][EPSILON].append(new_accept)

        start_states_stack.append(new_start)
        accept_states_stack.append(new_accept)

    @staticmethod
    def concatenate(
        start_state: State,
        accept_state: State,
        accept_states_stack: list[State],
        transition_table: dict[State, dict[Symbol, list[State]]],
    ) -> None:
        transition_table[start_state][EPSILON].append(accept_state)
        accept_states_stack.append(NFA.save_final)
        NFA.save_final = None

    # noinspection DuplicatedCode
    @staticmethod
    def kleene(
        start_state: State,
        accept_state: State,
        start_states_stack: list[State],
        accept_states_stack: list[State],
        transition_table: dict[State, dict[Symbol, list[State]]],
    ) -> None:
        new_start, new_accept = NFA.get_states_pair()

        transition_table[accept_state][EPSILON].append(start_state)
        transition_table[new_start][EPSILON].append(new_accept)
        transition_table[new_start][EPSILON].append(start_state)
        transition_table[accept_state][EPSILON].append(new_accept)

        start_states_stack.append(new_start)
        accept_states_stack.append(new_accept)

    @staticmethod
    def compute_symbol_set(postfix_regexp) -> set[Symbol]:
        return set(postfix_regexp) - set(PRECEDENCE.keys())

    @staticmethod
    def regexp_to_nfa(
        postfix_regexp: str,
    ) -> tuple[
        set[State],
        set[Symbol],
        State,
        State,
        dict[State, dict[Symbol, list[State]]],
    ]:
        start_states_stack: list[State] = []
        accept_states_stack: list[State] = []
        transition_table: dict[State, dict[Symbol, list[State]]] = defaultdict(
            lambda: defaultdict(list)
        )
        symbols: set[Symbol] = NFA.compute_symbol_set(postfix_regexp)
        final_state: State = EMPTY_STATE
        for i, c in enumerate(postfix_regexp):
            if c in symbols:
                frm, to = NFA.get_states_pair()
                transition_table[frm][c].append(to)
                start_states_stack.append(frm)
                accept_states_stack.append(to)

            elif c == UNION:
                lower_start, upper_start = (
                    start_states_stack.pop(),
                    start_states_stack.pop(),
                )
                lower_accept, upper_accept = (
                    accept_states_stack.pop(),
                    accept_states_stack.pop(),
                )
                NFA.union(
                    lower_start,
                    upper_start,
                    lower_accept,
                    upper_accept,
                    start_states_stack,
                    accept_states_stack,
                    transition_table,
                )

            elif c == CONCATENATION:
                NFA.save_final = accept_states_stack.pop()
                NFA.concatenate(
                    accept_states_stack.pop(),
                    start_states_stack.pop(),
                    accept_states_stack,
                    transition_table,
                )

            elif c == KLEENE_CLOSURE:
                NFA.kleene(
                    start_states_stack.pop(),
                    accept_states_stack.pop(),
                    start_states_stack,
                    accept_states_stack,
                    transition_table,
                )
            if i == (len(postfix_regexp) - 1):
                final_state: State = accept_states_stack.pop()
                final_state.is_accepting = True
                if EPSILON in symbols:
                    symbols.remove(EPSILON)

        initial_state = NFA.get_initial_state(start_states_stack)
        states = NFA.compute_states_set(transition_table)
        return states, symbols, initial_state, final_state, transition_table

    @staticmethod
    def compute_states_set(transition_table) -> set[State]:
        states = set()
        for s1, table in transition_table.items():
            states.add(s1)
            for s2s in table.values():
                if is_iterable(s2s):
                    for s2 in s2s:
                        states.add(s2)
                else:
                    states.add(s2s)
        return states

    def __repr__(self):
        return (
            f"NFA(states={self.states}, "
            f"symbols={self.symbols}, "
            f"start_state={self.start_state}, "
            f"accept_states={self.accept_states}) "
        )

    @staticmethod
    def get_initial_state(start_states_stack: list[State]):
        # after processing, start_states_stack should only have one item
        assert len(start_states_stack) == 1
        start_state: State = start_states_stack.pop()
        start_state.is_start = True
        return start_state

    def draw_with_graphviz(self):
        dot = graphviz.Digraph(
            self.__class__.__name__ + DFAState.max_label,
            format="pdf",
            engine="circo",
        )
        dot.attr("node", shape="circle")

        for symbol, s1, s2 in self.transitions:
            if s1.is_start:
                dot.node(
                    str(s1.state_id),
                    color="green",
                    shape="doublecircle" if s1.is_accepting else "circle",
                )
                dot.node("start", shape="none")
                dot.edge("start", f"{s1.state_id}", arrowhead="vee")
            else:
                dot.node(
                    f"{s1.state_id}",
                    shape="doublecircle" if s1.is_accepting else "circle",
                )

            dot.node(
                f"{s2.state_id}", shape="doublecircle" if s2.is_accepting else "circle"
            )
            dot.edge(str(s1.state_id), str(s2.state_id), label=symbol)

        dot.render(view=True)

    def _one_epsilon_closure_helper(self, s0: State, seen: set):
        seen.add(s0)
        closure = self.transition_table[s0][EPSILON][:]  # need to do a copy
        subs = []
        for s in closure:
            if (
                s not in seen
            ):  # to prevent infinite recursion when we encounter cycles in the NFA
                subs.append(self._one_epsilon_closure_helper(s, seen))
        return [s0] + closure + reduce(add, subs, [])

    def _one_epsilon_closure(self, s0: State) -> list[State]:
        return self._one_epsilon_closure_helper(s0, set())

    def epsilon_closure(self, states: Iterable):
        return frozenset(
            reduce(add, (self._one_epsilon_closure(state) for state in states), [])
        )

    def move(self, states: Iterable, symbol: Symbol) -> frozenset[State]:
        return frozenset(
            reduce(add, (self.transition_table[state][symbol] for state in states), [])
        )

    def find_state(self, state_id: int) -> Optional[State]:
        for state in self.states:
            if state_id == state.state_id:
                return state
        return None

    def get_dfa_state(self, from_states):
        dfa_from = DFAState(from_states=from_states)
        if self.start_state in dfa_from.from_states:
            dfa_from.is_start = True
        for accept_state in self.accept_states:
            if accept_state in dfa_from.from_states:
                dfa_from.is_accepting = True
                break
        return dfa_from

    def compute_transitions_for_dfa_state(
        self,
        dfa_from: DFAState,
        transition_table: dict[DFAState, dict[Symbol, DFAState]],
        seen: set[frozenset],
        stack: list[DFAState],
        accept_states: set[State],
    ):
        # what is the epsilon closure of the dfa_states
        eps = self.epsilon_closure(dfa_from.from_states)
        d = self.get_dfa_state(eps)
        if d.is_accepting:
            accept_states.add(d)
        # next we want to see which states are reachable from each of the states in the epsilon closure
        for symbol in self.symbols:
            next_states_set = self.epsilon_closure(self.move(eps, symbol))
            # new DFAState
            df = self.get_dfa_state(next_states_set)
            transition_table[d][symbol] = df
            if next_states_set not in seen:
                seen.add(next_states_set)
                stack.append(df)
        return d

    @staticmethod
    def clean_up_empty_sets(transition_table: dict[DFAState, dict[Symbol, DFAState]]):
        pruned_transition_table = defaultdict(dict)
        for start_state, table in transition_table.items():
            for symbol, end_state in table.items():
                if end_state.from_states:
                    pruned_transition_table[start_state][symbol] = end_state
        return pruned_transition_table

    @staticmethod
    def find_indistinguishable_pairs(states: set[State]) -> list[tuple[State, State]]:
        state_pairs = combinations(states, 2)
        return list(filter(lambda pair: DFA.states_eq(*pair), state_pairs))


def subset_construction(nfa: NFA):
    transitions_table: dict[DFAState, dict[Symbol, DFAState]] = defaultdict(dict)
    s0 = DFAState(from_states=frozenset({nfa.start_state}))
    accept_states: set[State] = set()
    seen = set()
    stack = [s0]

    to_explore = stack.pop()
    initial_state = nfa.compute_transitions_for_dfa_state(
        to_explore, transitions_table, seen, stack, accept_states
    )
    initial_state.is_start = True

    while stack:
        to_explore = stack.pop()
        nfa.compute_transitions_for_dfa_state(
            to_explore, transitions_table, seen, stack, accept_states
        )
    transitions_table = nfa.clean_up_empty_sets(transitions_table)
    states = NFA.compute_states_set(transitions_table)
    return states, nfa.symbols, transitions_table, initial_state, accept_states


def is_iterable(maybe_iterable):
    try:
        iter(maybe_iterable)
        return True
    except TypeError:
        return False


class DFA(NFA):
    def __init__(
        self,
        states: set[State],
        symbols: set[Symbol],
        transition_table: Callable[[State, Symbol], State]
        | dict[State, dict[Symbol, State]],
        start_state: State,
        accepting_states: frozenset[State],
    ):
        super().__init__(
            states, symbols, transition_table, start_state, accepting_states
        )
        self.states = states
        self.symbols = symbols
        self.transition_table = transition_table
        self.start_state = start_state
        self.accept_states = accepting_states

    @staticmethod
    def from_regexp(regexp: str) -> "NFA":
        return DFA(*subset_construction(NFA.from_regexp(regexp)))

    @staticmethod
    def from_nfa(nfa):
        return DFA(*subset_construction(nfa))

    @staticmethod
    def distinguish(
        states: tuple[tuple[State, State]],
        distinguish_function: Callable[[State, State], bool],
    ) -> dict[tuple[State, State], bool]:
        return {
            states_pair: distinguish_function(*states_pair) for states_pair in states
        }

    @staticmethod
    def collapse(ds: list[frozenset[State]]) -> set[State]:
        collapsed = set()
        for d in ds:
            for state in d:
                collapsed.add(state)
        return collapsed

    @staticmethod
    def reversed_transition_table(
        transition_table: dict[State, dict[Symbol, State]]
    ) -> dict[State, dict[Symbol, list[State]]]:
        ret = defaultdict(lambda: defaultdict(list))
        for start_state, table in transition_table.items():
            for symbol, end_state in table.items():
                ret[end_state][symbol].append(start_state)
        return ret

    @staticmethod
    def equivalence_partition(iterable, relation):
        """Partitions a set of objects into equivalence classes

        Args:
            iterable: collection of objects to be partitioned
            relation: equivalence relation. I.e. relation(o1, o2) evaluates to True
                if and only if o1 and o2 are equivalent

        Returns: classes, partitions
            classes: A sequence of sets. Each one is an equivalence class
            partitions: A dictionary mapping objects to equivalence classes
        """

        classes = []
        for o in iterable:  # for each object
            # find the class it is in
            found = False
            for c in classes:
                if relation(next(iter(c)), o):  # is it equivalent to this class?
                    c.add(o)
                    found = True
                    break
            if not found:  # it is in a new class
                classes.append({o})
        return set(map(frozenset, classes))

    def populate_in_dist(self) -> dict[State, dict[State, bool]]:
        in_dist = defaultdict(dict)
        for p, q in combinations(self.states, 2):
            in_dist[min(p, q)][max(p, q)] = p.is_accepting ^ q.is_accepting
        for p in self.states:
            in_dist[p][p] = False

        changed = True
        while changed:
            changed = False
            for _p, _q in combinations(self.states, 2):
                p, q = min(_p, _q), max(_p, _q)
                if not in_dist[p][q]:
                    for a in self.symbols:
                        k, m = self.make_transition(p, a), self.make_transition(q, a)
                        if in_dist[min(k, m)][max(k, m)]:
                            in_dist[p][q] = True  # distinguishable
                            changed = True

        return in_dist

    def get_new_states(self, in_dist: dict[State, dict[State, bool]]) -> list[DFAState]:
        def relation(p, q):
            return not in_dist[min(p, q)][max(p, q)]

        equivalence_classes = self.equivalence_partition(self.states, relation)

        new_states: list[DFAState] = list(
            map(
                self.get_dfa_state,
                filter(
                    lambda equivalence_class: len(equivalence_class) > 1,
                    equivalence_classes,
                ),
            )
        )
        return new_states

    def minimize(self):
        # if in_dist[p][q] == True, (p, q) are distinguishable

        in_dist = self.populate_in_dist()
        new_states = self.get_new_states(in_dist)
        rev_trans_table = self.reversed_transition_table(self.transition_table)
        new_trans = self.transition_table.copy()

        for new_state in new_states:
            for from_state in new_state.from_states:
                for symbol, source_states in rev_trans_table[from_state].items():
                    for source_state in source_states:
                        new_trans[source_state][symbol] = new_state

            new_states_iterable = iter(new_state.from_states)
            v = next(new_states_iterable)
            new_trans[new_state] = new_trans[v]
            new_trans.pop(v)

            for v in new_states_iterable:
                new_trans.pop(v)

        (starting_state,) = tuple(filter(lambda s: s.is_start, new_trans))
        accepting_states = frozenset(filter(lambda s: s.is_accepting, new_trans))

        return DFA(
            set(new_trans.keys()),
            self.symbols.copy(),
            new_trans,
            starting_state,
            accepting_states,
        )
