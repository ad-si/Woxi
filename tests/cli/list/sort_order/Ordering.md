# `Ordering`

Returns the permutation that sorts a list.

```scrut
$ wo 'Ordering[{30, 10, 20}]'
{2, 3, 1}
```

Asking for more positions than the list has is reported as a `Take` failure.

```scrut
$ wo 'Ordering[{3, 1, 2}, 5]'

Take::take: Cannot take positions 1 through 5 in {2, 3, 1}.
Ordering[{3, 1, 2}, 5]
```
