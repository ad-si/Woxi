# `Log`

Natural logarithm — `Log[b, x]` gives the logarithm of `x` in base `b`.

```scrut
$ wo 'Log[E]'
1
```

```scrut
$ wo 'Log[10, 100]'
2
```

The result collapses to a rational exponent for reciprocal and fractional
powers of the base.

```scrut
$ wo 'Log[2, 1/8]'
-3
```

```scrut
$ wo 'Log[4, 1/2]'
-1/2
```
