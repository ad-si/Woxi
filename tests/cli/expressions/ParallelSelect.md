# `ParallelSelect`

Parallel version of Select (evaluated sequentially).

```scrut
$ wo 'ParallelSelect[{1, 2, 3, 4}, EvenQ]'
{2, 4}
```

### Keeping only the first `n` matches

```scrut
$ wo 'ParallelSelect[Range[10], # > 3 &, 2]'
{4, 5}
```
