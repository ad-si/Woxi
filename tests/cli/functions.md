# Functions

## Nested Function Application

```scrut
$ wo 'Plus[Divide[6, 2], Abs[-5]]'
8
```


## Anonymous Identity Function

```scrut
$ wo '#&[1]'
1
```


### Function Application

```scrut
$ wo '#^2 &[{1, 2, 3}]'
{1, 4, 9}
```


## `/@` (Map)

Apply a function to each element of a list.

```scrut
$ wo 'Sign /@ {7, -2, 0, -5}'
{1, -1, 0, -1}
```

```scrut
$ wo '#^2& /@ {1, 2, 3}'
{1, 4, 9}
```

```scrut
$ wo 'Sin@(Pi/2)'
1
```

```scrut
$ wo '(Pi/2) // Sin'
1
```


## Define And Use A Function

```scrut
$ wo 'Double[x_] := x * 2; Double[5]'
10
```

```scrut
$ wo 'Double[x_] := x * 2; Double[Sin[Pi/2]]'
2
```

```scrut
$ wo 'Double[x_] := x * 2; Double @ Sin @ (Pi/2)'
2
```

```scrut
$ wo 'Double[x_] := x * 2; (Pi/2) // Sin // Double'
2
```


## `Apply` (`@@`)

Replaces the head of an expression with a function.

```scrut
$ wo 'f @@ {1, 2, 3}'
f[1, 2, 3]
```


## `Fold`

Applies a function cumulatively to elements of a list,
starting with an initial value.

```scrut
$ wo 'Fold[Plus, 0, {1, 2, 3}]'
6
```


## `FoldList`

Like [`Fold`](#fold), but returns a list of intermediate results.

```scrut
$ wo 'FoldList[Plus, 0, {1, 2, 3}]'
{0, 1, 3, 6}
```


## `Nest`

Applies a function repeatedly to an expression.

```scrut
$ wo 'Nest[f, x, 3]'
f[f[f[x]]]
```


## `NestList`

Like [`Nest`](#nest), but returns a list of intermediate results.

```scrut
$ wo 'NestList[f, x, 3]'
{x, f[x], f[f[x]], f[f[f[x]]]}
```


## `NestWhile`

Applies a function repeatedly while a test returns True.

```scrut
$ wo 'NestWhile[# + 1 &, 0, # < 5 &]'
5
```

```scrut
$ wo 'NestWhile[# / 2 &, 64, EvenQ]'
1
```


## `NestWhileList`

Like NestWhile, but returns a list of all intermediate results.

```scrut
$ wo 'NestWhileList[# + 1 &, 0, # < 5 &]'
{0, 1, 2, 3, 4, 5}
```

```scrut
$ wo 'NestWhileList[# / 2 &, 64, EvenQ]'
{64, 32, 16, 8, 4, 2, 1}
```


## `Through`

Applies a list of functions to an argument.

```scrut
$ wo 'Through[{Sin, Cos}, 0]'
{Sin, Cos}
```

```scrut
$ wo 'Through[{Abs, Sign}, -5]'
{Abs, Sign}
```


## `TakeLargest`

Returns the n largest elements from a list.

```scrut
$ wo 'TakeLargest[{3, 1, 4, 1, 5, 9, 2, 6}, 3]'
{9, 6, 5}
```

```scrut
$ wo 'TakeLargest[{5, 2, 8, 1}, 2]'
{8, 5}
```


## `TakeSmallest`

Returns the n smallest elements from a list.

```scrut
$ wo 'TakeSmallest[{3, 1, 4, 1, 5, 9, 2, 6}, 3]'
{1, 1, 2}
```

```scrut
$ wo 'TakeSmallest[{5, 2, 8, 1}, 2]'
{1, 2}
```


## `ArrayDepth`

Returns the depth of a nested list.

```scrut
$ wo 'ArrayDepth[{1, 2, 3}]'
1
```

```scrut
$ wo 'ArrayDepth[{{1, 2}, {3, 4}}]'
2
```

```scrut
$ wo 'ArrayDepth[{{{1}}}]'
3
```

```scrut
$ wo 'ArrayDepth[5]'
0
```


## `DateString`

```scrut
$ wo 'StringStartsQ[DateString[Now, "ISODateTime"], "2026-"]'
True
```
