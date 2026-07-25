# `CellularAutomaton`

Generates a cellular automaton evolution.

```scrut
$ wo 'CellularAutomaton[90, {{1}, 0}, 2]'
{{0, 0, 1, 0, 0}, {0, 1, 0, 1, 0}, {1, 0, 0, 0, 1}}
```

The step specification `{{t}}` returns just the state at step `t`, wrapped in
a list (`{t}` behaves like the bare `t` form, giving all steps `0` through `t`):

```scrut
$ wo 'CellularAutomaton[30, {{1}, 0}, {{3}}]'
{{1, 1, 0, 1, 1, 1, 1}}
```

Two-dimensional rules take a weight matrix and a range specification.
`{{t1, t2, dt}}` selects the states at steps `t1` through `t2` in
increments of `dt`:

```scrut
$ wo 'ArrayPlot /@ CellularAutomaton[{942, {2, {{0, 2, 0}, {2, 1, 2}, {0, 2, 0}}}, {1, 1}}, {{{1}}, 0}, {{10, 30, 10}}]'
{-Graphics-, -Graphics-, -Graphics-}
```

A second element in the step specification restricts the cells returned.
`All` keeps every affected cell, while `{x1, x2}` names offsets relative to
the first cell of the initial condition:

```scrut
$ wo 'CellularAutomaton[90, {{1}, 0}, {3, {-2, 2}}]'
{{0, 0, 1, 0, 0}, {0, 1, 0, 1, 0}, {1, 0, 0, 0, 1}, {0, 1, 0, 1, 0}}
```

Without a step count only one step runs, and just the new state comes back —
as a `{cells, {background}}` pair when the initial condition has a background:

```scrut
$ wo 'CellularAutomaton[30, {{1}, 0}]'
{{1, 1, 1}, {0}}
```

The operator form does the same:

```scrut
$ wo 'CellularAutomaton[30][{0, 0, 1, 0, 0}]'
{0, 1, 1, 1, 0}
```

```scrut
$ wo 'CellularAutomaton[x, {{1}, 0}, 3]'

CellularAutomaton::nspecnl: Rule specification x should be an Integer, a List, a pure Boolean function, a String or an Association.
CellularAutomaton[x, {{1}, 0}, 3]
```
