# `Partition`


Breaks a list into smaller sublists.

```scrut
$ wo 'Partition[{1, 2, 3, 4}, 2]'
{{1, 2}, {3, 4}}
```

A sixth argument is the head every block is wrapped in instead of `List`:

```scrut
$ wo 'Partition[{1, 2, 3, 4}, 2, 2, 1, {}, f]'
{f[1, 2], f[3, 4]}
```
