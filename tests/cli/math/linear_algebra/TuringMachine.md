# `TuringMachine`

Simulate a Turing machine from a rule number and initial conditions.

```scrut
$ wo 'TuringMachine[2506, {{1}, {0, 1, 0}}, 0]'
{{{1, 1, 0}, {0, 1, 0}}}
```

The state may be named directly, and each result pairs
`{state, position, offset}` with the tape:

```scrut
$ wo 'TuringMachine[2506, {1, {0, 0, 0, 0}}, 3]'
{{{1, 1, 0}, {0, 0, 0, 0}}, {{2, 2, 1}, {1, 0, 0, 0}}, {{1, 1, 0}, {1, 1, 0, 0}}, {{2, 4, -1}, {0, 1, 0, 0}}}
```

A `{cells, background}` tape is infinite: the head never wraps, and the
reported tape is the region the run actually visited.

```scrut
$ wo 'TuringMachine[2506, {1, {{}, 0}}, 4]'
{{{1, 3, 0}, {0, 0, 0, 0}}, {{2, 4, 1}, {0, 0, 1, 0}}, {{1, 3, 0}, {0, 0, 1, 1}}, {{2, 2, -1}, {0, 0, 0, 1}}, {{1, 1, -2}, {0, 1, 0, 1}}}
```

Without a step count only one step runs; the operator form does the same:

```scrut
$ wo 'TuringMachine[2506][{1, {{}, 0}}]'
{{2, 2, 1}, {{1, 0}, 0}}
```
