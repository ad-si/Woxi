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
