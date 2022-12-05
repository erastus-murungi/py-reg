# PyReg

A simple NFA-based regular expression engine.

### Supports:
  1. ``|`` Alternation
  2. ``*`` Zero or More
  3. ``?`` Zero or One
  4. ``+`` One or More
  5. ``{x, y?}`` Character repetition
  6. ``[x-y]`` Character classes
  7. ``()`` Non-capturing grouping
  8. ``.`` Match any character operator


### Extras
An NFA to DFA implementation via subset construction.
A resulting DFA which can be minimized.
