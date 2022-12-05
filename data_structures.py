from collections import defaultdict
from symbol import Epsilon, CompoundSymbol, Symbol
from typing import Callable, MutableMapping


class UnionFind:
    """A simple implementation of a disjoint-set data structure.
    The amortized running time is O(m ⍺(n)) for m disjoint-set operations on n elements, where
    ⍺(n) is the inverse Ackermann function .⍺(n) grows extremely slowly and can be assumed to be ⩽ 5 for
    all practical purposes.
    Operations:
        MAKE-SET(x) – creates a new set with one element {x}.
        UNION(x, y) – merge into one set the set that contains element x and the set that contains element y (x and y are in different sets). The original sets will be destroyed.
        FIND-SET(x) – returns the representative or a pointer to the representative of the set that contains element x.
    Applications of UnionFind include:
        1. Kruskal’s algorithm for MST.
        2. They are useful in applications like “Computing the shorelines of a terrain,”
            “Classifying a set of atoms into molecules or fragments,” “Connected component labeling in image analysis,” and others.[1]
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
                {self[x] for x in objects}, key=lambda r: self.weights[r], reverse=True
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


class SymbolDispatchedMapping(MutableMapping):
    def __init__(self, default=None):
        self.default_factory = default
        self.sorted_map: list[tuple[CompoundSymbol, Callable]] = []
        if default is None:
            self.hash_map = {}
        else:
            self.hash_map: defaultdict[Symbol, Callable] = defaultdict(default)

    def __setitem__(self, symbol, v) -> None:
        if isinstance(symbol, CompoundSymbol):
            for i in range(len(self.sorted_map)):
                sym, value = self.sorted_map[i]
                if sym.match(symbol):
                    self.sorted_map[i] = (symbol, v)
                    return
            self.sorted_map.append((symbol, v))
        else:
            self.hash_map[symbol] = v

    def __delitem__(self, __v) -> None:
        raise NotImplementedError

    def __getitem__(self, symbol: Symbol):
        if isinstance(symbol, CompoundSymbol):
            for sym, value in self.sorted_map:
                if sym.match(symbol):
                    return value
            if self.default_factory is not None:
                value = self.default_factory()
                self.sorted_map.append((symbol, value))
                return value
        return self.hash_map.__getitem__(symbol)

    def __len__(self) -> int:
        return len(self.sorted_map) + len(self.hash_map)

    def __iter__(self):
        yield from map(lambda tup: tup[0], self.sorted_map)
        yield from self.hash_map

    def items(self):
        yield from self.sorted_map
        yield from self.hash_map.items()

    def items_non_epsilon(self):
        for symbol, val in self.items():
            if not symbol.match(Epsilon):
                yield symbol, val

    def clear(self) -> None:
        self.sorted_map = []
        if self.default_factory is None:
            self.hash_map = {}
        else:
            self.hash_map = defaultdict(self.default_factory)

    def update(self, m, **kwargs) -> None:
        for k, v in m.items():
            self[k] = v

    def match_atom(self, char, default):
        for sym, value in self.sorted_map:
            if sym == char:
                return value
        return self.hash_map.get(char, default)

    def __repr__(self):
        return f"{self.__class__.__name__}(compound={self.sorted_map}, simple={self.hash_map})"
