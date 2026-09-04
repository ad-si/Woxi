# `ParallelSelect`

Parallel version of Select (evaluated sequentially).

```scrut
$ wo 'ParallelSelect[{1, 2, 3, 4}, EvenQ]'
{2, 4}
```

### Keeping only the first `n` matches

Capping the number of matches is the one argument form `Select` has no
parallel implementation for, so it falls back to a sequential evaluation and
says so:

```scrut
$ wo 'ParallelSelect[{1, 2, 3, 4, 5}, # > 3 &, 2]'

ParallelSelect::nopar1: Select[{1, 2, 3, 4, 5}, #1 > 3 & , 2] cannot be parallelized; proceeding with sequential evaluation.
{4, 5}
```
