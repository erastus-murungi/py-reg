from collections import defaultdict
from enum import IntFlag, auto
from typing import Generic, NamedTuple, TypeVar

T = TypeVar("T")


class Fragment(NamedTuple, Generic[T]):
    start: T
    end: T

    @staticmethod
    def duplicate(item: T) -> "Fragment[T]":
        return Fragment(item, item)


class RegexFlag(IntFlag):
    NO_BACKTRACK = auto()
    NOFLAG = auto()
    IGNORECASE = auto()
    MULTILINE = auto()
    DOTALL = auto()  # make dot match newline
    FREESPACING = auto()
    OPTIMIZE = auto()
    DEBUG = auto()

    def should_optimize(self) -> bool:
        return bool(self & RegexFlag.OPTIMIZE)


class UnionFind:
    """A simple implementation of a disjoint-set data structure.
    The amortized running time is O(m ⍺(n)) for m disjoint-set operations on n elements, where
    ⍺(n) is the inverse Ackermann function .⍺(n) grows extremely slowly and can be assumed to be ⩽ 5 for
    all practical purposes.
    Operations:
        MAKE-SET(x) – creates a new set with one element {x}.
        UNION(x, y) – merge into one set the set that contains element x and the
                        set that contains element y (x and y are in different sets).
                         The original sets will be destroyed.
        FIND-SET(x) – returns the representative or a pointer to the representative of the set that contains element x.
    Applications of UnionFind include:
        1. Kruskal’s algorithm for MST.
        2. They are useful in applications like “Computing the shorelines of a terrain,”
            “Classifying a set of atoms into molecules or fragments,” “Connected component labeling in image analysis,”
                and others.[1]
        3. Labeling connected components.
        4. Random maze generation and exploration.
        5. Alias analysis in compiler theory.
        6. Maintaining the connected components of an undirected-graph, when the edges are being added dynamically.
        7. Strategies for games: Hex and Go.
        8. Tarjan's offline Least common ancestor algorithm.
        9. Cycle detection in undirected graph.
        10. Equivalence of finite state automata

    Reference:
        1.  Cormen, Leiserson, Rivest, Stein,. "Chapter 21: Data structures for Disjoint Sets".
            Introduction to Algorithms (Third ed.). MIT Press. pp. 571–572. ISBN 978-0-262-03384-8.
        2.  https://www.topcoder.com/community/competitive-programming/tutorials/disjoint-set-data-structures/
        3.  https://en.wikipedia.org/wiki/Disjoint-set_data_structure
        4.  https://www.cs.upc.edu/~mjserna/docencia/grauA/T19/Union-Find.pdf

    """

    def __init__(self, items=()):
        # MAKE-SET()
        # we don't need to store the elements, instead we can hash them, and since each element is unique, then
        # hashes won't collide. I will use union by size instead of union by rank
        # using union by rank needs more careful handling in the union of multiple items

        self.parents = {}
        self.weights = {}

        for item in items:
            self.parents[item] = item
            self.weights[item] = 1

    def __getitem__(self, item):
        # FIND-SET()
        if item not in self.parents:
            self.parents[item] = item  # MAKE-SET()
            self.weights[item] = 1
            return item
        else:
            # store nodes in the path leading to the root(representative) for later updating
            # this is the path-compression step
            path = [item]
            root = self.parents[item]
            while root != path[-1]:
                path.append(root)
                root = self.parents[root]
            for node in path:
                self.parents[node] = root
            return root

    def union(self, *objects):
        """Find the sets containing the objects and merge them all."""
        # Find the heaviest root according to its weight.
        roots = iter(
            sorted(
                {self[x] for x in objects}, key=lambda h: self.weights[h], reverse=True
            )
        )
        try:
            heaviest = next(roots)
        except StopIteration:
            return

        for r in roots:
            self.weights[heaviest] += self.weights[r]
            self.parents[r] = heaviest

    def __iter__(self):
        return iter(self.parents)

    def _groups(self):
        one_to_many = defaultdict(set)
        for v, k in self.parents.items():
            one_to_many[k].add(v)
        return dict(one_to_many)

    def to_sets(self):
        for x in self.parents.keys():
            _ = self[x]  # Evaluated for side effect only

        yield from self._groups().values()

    def __str__(self):
        return str(list(self.to_sets()))
