# `DeleteDuplicates`

Removes duplicate elements from a list.

```scrut
$ wo 'DeleteDuplicates[{a, b, a, c, b}]'
{a, b, c}
```

```scrut
$ wo 'DeleteDuplicates[{1, 2, 1, 3, 2}]'
{1, 2, 3}
```

```scrut
$ wo 'DeleteDuplicates[{1, 1, 1}]'
{1}
```

```scrut
$ wo 'DeleteDuplicates[{a}]'
{a}
```

```scrut
$ wo 'DeleteDuplicates[{}]'
{}
```

On an association, duplicates are removed by value, keeping the first key.

```scrut
$ wo 'DeleteDuplicates[<|a -> 1, b -> 1, c -> 2|>]'
<|a -> 1, c -> 2|>
```
