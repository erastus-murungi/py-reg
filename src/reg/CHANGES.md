### Improvement from table filling method of DFA minimization

Former algorithm:

```python
    def gen_equivalence_states_table_filling(
        transitions, accepting_states, alphabet
    ) -> Iterator[set[State]]:
        """
        Myhill-Nerode Theorem
        https://www.cs.scranton.edu/~mccloske/courses/cmps364/dfa_minimize.html
        """

        def transition(state: State, target_matcher: Matcher) -> State | tuple[State, ...]:
            for matcher, end in transitions[state]:
                if matcher == target_matcher:
                    return end
            return ""

        states = transitions.keys()
        # a state is indistinguishable from itself
        indistinguishable = {(p, p) for p in states}

        for p, q in combinations(states, 2):
            # a pair of states are maybe indistinguishable
            # if they are both accepting or both non-accepting
            # we use min max to provide an ordering based on the labels
            p, q = minmax(p, q)
            if (p in accepting_states) == (q in accepting_states):
                indistinguishable.add((p, q))

        union_find = UnionFind(states)

        changed = True
        while changed:
            changed = False
            removed = set()
            for p, q in indistinguishable:
                if p == q:
                    continue
                # if two states are maybe indistinguishable, then do some more work to prove they are actually
                # indistinguishable
                for a in alphabet:
                    km = minmax(transition(p, a), transition(q, a))
                    if km != ("", "") and km not in indistinguishable:
                        removed.add((p, q))
                        changed = True
            indistinguishable = indistinguishable - removed

        for p, q in indistinguishable:
            union_find.union(p, q)

        return union_find.to_sets()
```