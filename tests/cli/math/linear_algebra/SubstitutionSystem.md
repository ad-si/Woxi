# `SubstitutionSystem`

Applies substitution rules iteratively, returning every state from the
initial condition through step `t`.

Strings substitute character by character:

```scrut
$ wo 'SubstitutionSystem[{"A" -> "AB", "B" -> "A"}, "A", 4]'
{A, AB, ABA, ABAAB, ABAABABA}
```

Lists substitute element by element:

```scrut
$ wo 'SubstitutionSystem[{0 -> {0, 1}, 1 -> {1, 0}}, {0}, 3]'
{{0}, {0, 1}, {0, 1, 1, 0}, {0, 1, 1, 0, 1, 0, 0, 1}}
```

When a rule replaces a cell with a matrix the system is two-dimensional:
every cell expands into a block and the blocks tile, so an `m`×`n` grid
whose cells expand to `p`×`q` becomes `(m p)`×`(n q)`.

```scrut
$ wo 'SubstitutionSystem[{1 -> {{1, 0}, {1, 1}}, 0 -> {{0, 0}, {0, 0}}}, {{1}}, 2]'
{{{1}}, {{1, 0}, {1, 1}}, {{1, 0, 0, 0}, {1, 1, 0, 0}, {1, 0, 1, 0}, {1, 1, 1, 1}}}
```

Without a step count only one step runs, and just the new state comes back.
The operator form does the same:

```scrut
$ wo 'SubstitutionSystem[{1 -> {1, 0}, 0 -> {1}}][{1}]'
{1, 0}
```
