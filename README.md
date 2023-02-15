# PyReg

2 implementations of regular expression engines

One is a backtracking regex engine implemented using a tagged NFA.
The second is a non-backtracking engine implemented using a virtual machine, referered to as Pike's VM by Russ Cox.

Insipiration is drawn heavily from the [excellent series](https://swtch.com/~rsc/regexp/) of articles by Russ Cox.

Although tested, these implementations are not meant for production use but rather for learning purposes. I implemented them in Python for ease of understanding. I am planning to port them to another language such as Rust or C for perfomance.

Other references include:
  1. [An additional non-backtracking RegExp engine](https://v8.dev/blog/non-backtracking-regexp)
  2. [Regex Benchmarks](https://github.com/mariomka/regex-benchmark)

![Tests](https://github.com/erastus-murungi/py-reg/actions/workflows/python-app.yml/badge.svg)

### Supports:
  1. ``|`` Alternation
  2. ``*`` Zero or More
  3. ``?`` Zero or One
  4. ``+`` One or More
  5. ``{x, y?}`` Character repetition
  6. ``[x-y]`` Character classes
  7. ``(?:ab)`` Non-capturing grouping
  8. ``(ab)`` Capturing groups
  9. ``.`` Match any character operator
  10. Lazy ops: ``*?``, ``+?``, ``??``, ``{x, y?}?`` 
  11. Anchors such as ``$, ^, \b, \B \Z \z \A``
  12. Regex Flags
  13. Inline modifiers: (?m, ?i, ?s)

Features such as backreferences and lookahead assertions are not implemented because I cannot be implemented efficiently.


### Extras
An NFA to DFA implementation via subset construction.
A resulting DFA which can be minimized.
