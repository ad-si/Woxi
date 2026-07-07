# `CellularAutomaton`

Generates a cellular automaton evolution.

```scrut
$ wo 'CellularAutomaton[90, {{1}, 0}, 2]'
{{0, 0, 1, 0, 0}, {0, 1, 0, 1, 0}, {1, 0, 0, 0, 1}}
```

The step specification `{t}` returns only the state at step `t`:

```scrut
$ wo 'CellularAutomaton[30, {{1}, 0}, {3}]'
{1, 1, 0, 1, 1, 1, 1}
```

Two-dimensional rules take a weight matrix and a range specification.
`{{t1, t2, dt}}` selects the states at steps `t1` through `t2` in
increments of `dt`:

```scrut
$ wo 'ArrayPlot /@ CellularAutomaton[{942, {2, {{0, 2, 0}, {2, 1, 2}, {0, 2, 0}}}, {1, 1}}, {{{1}}, 0}, {{10, 30, 10}}]'
{-Graphics-, -Graphics-, -Graphics-}
```

```scrut
$ wo 'CellularAutomaton[x, {{1}, 0}, 3]'

CellularAutomaton::nspecnl: Rule specification x should be an Integer, a List, a pure Boolean function, a String or an Association.
CellularAutomaton[x, {{1}, 0}, 3]
```
