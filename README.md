# PyReg

A simple Backtracking NFA-based regular expression engine.

![Tests](https://github.com/erastus-murungi/py-reg/actions/workflows/python-app.yml/badge.svg)

### Supports:
  1. ``|`` Alternation
  2. ``*`` Zero or More
  3. ``?`` Zero or One
  4. ``+`` One or More
  5. ``{x, y?}`` Character repetition
  6. ``[x-y]`` Character classes
  7. ``()`` Non-capturing grouping
  8. ``.`` Match any character operator
  9. Lazy ops: ``*?``, ``+?``, ``??``, ``{x, y?}?`` 
  10. Anchors such as ``$, ^, \b, \B \Z \z \A``


### Extras
An NFA to DFA implementation via subset construction.
A resulting DFA which can be minimized.
