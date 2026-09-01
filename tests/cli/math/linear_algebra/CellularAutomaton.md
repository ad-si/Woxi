# `CellularAutomaton`

Generates a cellular automaton evolution.

```scrut
$ wo 'CellularAutomaton[90, {{1}, 0}, 2]'
{{0, 0, 1, 0, 0}, {0, 1, 0, 1, 0}, {1, 0, 0, 0, 1}}
```

The third argument is either a bare `t` or the list `{tspec, xspec[, yspec]}`,
whose leading element is always the tspec. So `{3}` is that list with
`tspec = 3` — every step `0` through `3`, like the bare form — while `{{3}}`
puts the tspec `{3}` there, which is step `3` alone, still wrapped in a list:

```scrut
$ wo 'CellularAutomaton[30, {{1}, 0}, {{3}}]'
{{1, 1, 0, 1, 1, 1, 1}}
```

One more layer of braces, `{{{t}}}`, returns that state bare:

```scrut
$ wo 'CellularAutomaton[30, {{1}, 0}, {{{3}}}]'
{1, 1, 0, 1, 1, 1, 1}
```

A two- or three-element tspec is a step range: `{t1, t2}` gives steps `t1`
through `t2`, and `{t1, t2, dt}` every `dt`-th of those. Written without the
extra braces the same numbers are space windows instead — `{1, 3}` is "steps
`0` through `1`, cells `0` through `3`":

```scrut
$ wo 'CellularAutomaton[90, {{1}, 0}, {{1, 3}}]'
{{0, 0, 1, 0, 1, 0, 0}, {0, 1, 0, 0, 0, 1, 0}, {1, 0, 1, 0, 1, 0, 1}}
```

```scrut
$ wo 'CellularAutomaton[90, {{1}, 0}, {1, 3}]'
{{1, 0, 0, 0}, {0, 1, 0, 0}}
```

Two-dimensional rules take a weight matrix and a range specification:

```scrut
$ wo 'ArrayPlot /@ CellularAutomaton[{942, {2, {{0, 2, 0}, {2, 1, 2}, {0, 2, 0}}}, {1, 1}}, {{{1}}, 0}, {{10, 30, 10}}]'
{-Graphics-, -Graphics-, -Graphics-}
```

`{tspec, xspec, yspec}` restricts a two-dimensional rule's rows and
columns the same way `{tspec, xspec}` restricts a one-dimensional rule's
cells; a two-dimensional rule given only an `xspec` windows its rows and
keeps every column. The bare `{{t}}` tspec is what `ArrayPlot` needs to plot
a state directly:

```scrut
$ wo 'CellularAutomaton[{942, {2, {{0, 2, 0}, {2, 1, 2}, {0, 2, 0}}}, {1, 1}}, {{{1}}, 0}, {{{1}}, All, All}]'
{{0, 1, 0}, {1, 1, 1}, {0, 1, 0}}
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
