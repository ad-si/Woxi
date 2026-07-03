# `ParallelDo`

Parallel version of `Do` (evaluated sequentially in Woxi).

It evaluates its body for each value of the iterator and returns `Null`:

```scrut
$ wo '{ParallelDo[i, {i, 3}], 42}'
{Null, 42}
```
