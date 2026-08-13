# `ParallelCases`

Parallel version of Cases (evaluated sequentially).

```scrut
$ wo 'ParallelCases[{1, a, 2, b, 3}, _Integer]'
{1, 2, 3}
```

### Cases with level specification

```scrut
$ wo 'ParallelCases[{{1, 2}, {3, 4}}, _Integer, {2}]'
{1, 2, 3, 4}
```
