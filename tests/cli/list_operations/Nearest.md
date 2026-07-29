# `Nearest`

Find the nearest elements in a list to a given value.

```scrut
$ wo 'Nearest[{1, 5, 10, 12}, 11]'
{10, 12}
```

A count limits how many of the closest elements are returned.

```scrut
$ wo 'Nearest[{1, 2, 3, 10}, 4, 2]'
{3, 2}
```

Strings are compared by edit distance.

```scrut
$ wo 'Nearest[{"cat", "car", "dog"}, "cot"]'
{cat}
```

An empty point set is unusable rather than a set in which nothing happens to
be near, so it is reported and the call is left as written:

```scrut
$ wo 'Nearest[{}, 1]'

Nearest::near1: {} is neither a list of real points nor a valid list of rules.
Nearest[{}, 1]
```

The same goes for data that is not a list of points at all:

```scrut
$ wo 'Nearest[5, 1]'

Nearest::near1: 5 is neither a list of real points nor a valid list of rules.
Nearest[5, 1]
```
