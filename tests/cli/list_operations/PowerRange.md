# `PowerRange`

Generate a list of values by successive multiplication.

```scrut
$ wo 'PowerRange[1, 100]'
{1, 10, 100}
```

Terms are kept while their magnitude stays inside the bound, and the factor
sets the direction: one above `1` grows towards the bound, one below shrinks
towards it.

```scrut
$ wo 'PowerRange[100, 1, 1/10]'
{100, 10, 1}
```

```scrut
$ wo 'PowerRange[1, 5, -2]'
{1, -2, 4}
```

Bounds that repeated multiplication can never stay inside are reported:

```scrut
$ wo 'PowerRange[-1, 5]'

PowerRange::range: Range specification in PowerRange[-1, 5] does not have appropriate bounds.
PowerRange[-1, 5]
```

```scrut
$ wo 'PowerRange[1, 5, 1]'

PowerRange::factor: Factor cannot be 1.
PowerRange[1, 5, 1]
```
