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

`Log[E^z]` unwraps a numeric complex exponent, reducing the imaginary part
into `(-Pi, Pi]`.

```scrut
$ wo 'Log[E^(2 + 3 I)]'
2 + 3*I
```

```scrut
$ wo 'Log[E^(5 I)]'
5*I - (2*I)*Pi
```
