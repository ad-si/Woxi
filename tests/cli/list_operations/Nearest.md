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

`DistanceFunction -> f` measures with `f[element, target]` instead of the
built-in metric, so a named metric picks the neighbour:

```scrut
$ wo 'Nearest[{{0, 0}, {3, 4}}, {0, 1}, DistanceFunction -> ManhattanDistance]'
{{0, 0}}
```

A pure function works too — negating the distance asks for the farthest element
instead of the nearest:

```scrut
$ wo 'Nearest[{1, 2, 3, 10}, 4, DistanceFunction -> (-Abs[#1 - #2] &)]'
{10}
```

It applies through the other forms as well, here with a count:

```scrut
$ wo 'Nearest[{1, 2, 3, 10}, 4, 2, DistanceFunction -> (Abs[#1 - #2] &)]'
{3, 2}
```

Any other option is accepted and changes nothing:

```scrut
$ wo 'Nearest[{1, 2, 3, 10}, 4, Method -> Automatic]'
{3}
```
