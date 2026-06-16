# `ReverseSort`

Sorts a list in descending order.

```scrut
$ wo 'ReverseSort[{3, 1, 4, 1, 5, 9}]'
{9, 5, 4, 3, 1, 1}
```

On an association, the pairs are sorted descending by value.

```scrut
$ wo 'ReverseSort[<|a -> 1, b -> 3, c -> 2|>]'
<|b -> 3, c -> 2, a -> 1|>
```
