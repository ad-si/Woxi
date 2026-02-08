# Symbolic Computing

```scrut
$ wo 'cow + 5'
5 + cow
```

```scrut
$ wo 'cow + 5 + 10'
15 + cow
```

```scrut
$ wo 'moo = cow + 5'
5 + cow
```

```scrut
$ wo 'D[x^n, x]'
n*x^(-1 + n)
```

```scrut
$ wo 'Integrate[x^2 + Sin[x], x]'
x^3/3 - Cos[x]
```


## Limits

```scrut
$ wo 'Limit[Sin[x]/x, x -> 0]'
1
```


## Series

```scrut
$ wo 'Series[Exp[x], {x, 0, 3}]'
SeriesData[x, 0, {1, 1, 1/2, 1/6}, 0, 4, 1]
```


## Apart

```scrut
$ wo 'Apart[1/(x^2 - 1)]'
1/(2*(-1 + x)) - 1/(2*(1 + x))
```


## Together

```scrut
$ wo 'Together[1/x + 1/y]'
(x + y)/(x*y)
```


## Cancel

```scrut
$ wo 'Cancel[(x^2 - 1)/(x - 1)]'
1 + x
```


## Collect

```scrut
$ wo 'Collect[x*y + x*z, x]'
x*(y + z)
```


## ExpandAll

```scrut
$ wo 'ExpandAll[x*(x + 1)^2]'
x + 2*x^2 + x^3
```
