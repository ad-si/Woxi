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


## `Run`

Executes a system command and returns the exit code.

```scrut
$ wo 'Run["echo hello"]'
hello
0
```

```scrut
$ wo 'Run["exit 0"]'
0
```

```scrut
$ wo 'Run["exit 1"]'
256
```


## Pattern Definitions with `SetDelayed`

Define a function using pattern matching.

```scrut
$ wo 'f[x_] := x^2; f[3]'
9
```

```scrut
$ wo 'f[x_] := x^2; f[5]'
25
```

```scrut
$ wo 'f[x_] := x + 1; f[10]'
11
```


## Optional Pattern Arguments

Optional arguments use the `x_:default` syntax to provide default values.

```scrut
$ wo 'f[x_:0] := x + 1; f[]'
1
```

```scrut
$ wo 'f[x_:0] := x + 1; f[5]'
6
```

### Optional with head constraint

```scrut
$ wo 'g[q_List: {}, n_] := {q, n}; g[5]'
{{}, 5}
```

```scrut
$ wo 'g[q_List: {}, n_] := {q, n}; g[{1, 2}, 3]'
{{1, 2}, 3}
```

### Multiple optional arguments

```scrut
$ wo 'f[a_:0, b_, c_:99] := {a, b, c}; f[5]'
{0, 5, 99}
```

```scrut
$ wo 'f[a_:0, b_, c_:99] := {a, b, c}; f[1, 5]'
{1, 5, 99}
```

```scrut
$ wo 'f[a_:0, b_, c_:99] := {a, b, c}; f[1, 5, 10]'
{1, 5, 10}
```


## `With`

`With` substitutes constant values into the body expression.

```scrut
$ wo 'With[{x = 5}, x + 1]'
6
```

```scrut
$ wo 'With[{x = 2, y = 3}, x + y]'
5
```

```scrut
$ wo 'With[{l = Length[{1,2,3}]}, l + 1]'
4
```


# Control Flow

## `If`

Conditional expression — covered in [boolean.md](boolean.md#if) as well.

```scrut
$ wo 'If[1 > 0, "yes", "no"]'
yes
```


## `For`

`For[init, test, step, body]` loop.

```scrut
$ wo 'For[i = 0; s = 0, i <= 10, i++, s = s + i]; s'
55
```


## `While`

`While[test, body]` loop.

```scrut
$ wo 'i = 0; While[i < 5, i++]; i'
5
```


## `Do`

`Do[expr, {iter}]` iterates `expr` for each value of the iterator.
The result is always `Null`.

```scrut
$ wo 'Do[Print[k], {k, 1, 3}]'
1
2
3
Null
```


## `Switch`

Multi-way branching — the first pattern that matches wins.

```scrut
$ wo 'Switch[2, 1, "one", 2, "two", _, "other"]'
two
```


## `Which`

Returns the first value whose test evaluates to `True`.

```scrut
$ wo 'Which[2 > 1, "a", True, "b"]'
a
```


## `Piecewise`

Defines a piecewise-defined expression from a list of
`{value, condition}` pairs.

```scrut
$ wo 'Piecewise[{{1, x > 0}}] /. x -> 1'
1
```


## `FixedPoint`

Applies a function repeatedly until the result stops changing,
or until `n` iterations have occurred.

```scrut
$ wo 'FixedPoint[# + 1 &, 0, 3]'
3
```


## `FixedPointList`

Like `FixedPoint` but returns the entire history of values.

```scrut
$ wo 'FixedPointList[#/2 &, 8, 3]'
{8, 4, 2, 1}
```


## `NestWhile`

Applies a function repeatedly while a predicate holds.

```scrut
$ wo 'NestWhile[#/2 &, 8, EvenQ]'
1
```


## `NestWhileList`

Like `NestWhile` but returns the entire history.

```scrut
$ wo 'NestWhileList[#/2 &, 8, EvenQ]'
{8, 4, 2, 1}
```


## `CompoundExpression`

`CompoundExpression[e1, e2, …]` (written `e1; e2; …`) evaluates
each argument in order and returns the last one.

```scrut
$ wo 'CompoundExpression[1, 2, 3]'
3
```


# Mutating Assignment

## `AddTo` (`+=`)

Increments a variable in place.

```scrut
$ wo 'x = 5; x += 3; x'
8
```


## `SubtractFrom` (`-=`)

Decrements a variable in place.

```scrut
$ wo 'x = 5; x -= 3; x'
2
```


## `TimesBy` (`*=`)

Multiplies a variable in place.

```scrut
$ wo 'x = 5; x *= 3; x'
15
```


## `DivideBy` (`/=`)

Divides a variable in place.

```scrut
$ wo 'x = 6; x /= 3; x'
2
```


## `Increment` (`x++`)

Post-increment — returns the old value, then adds 1.

```scrut
$ wo 'x = 5; x++'
5
```


## `Decrement` (`x--`)

Post-decrement — returns the old value, then subtracts 1.

```scrut
$ wo 'x = 5; x--'
5
```


## `PreIncrement` (`++x`)

Pre-increment — adds 1, then returns the new value.

```scrut
$ wo 'x = 5; ++x'
6
```


## `PreDecrement` (`--x`)

Pre-decrement — subtracts 1, then returns the new value.

```scrut
$ wo 'x = 5; --x'
4
```


# Scoping and Exceptions

## `Block`

Like `Module`, but temporarily rebinds *global* symbols rather than
introducing fresh locals. Changes are undone when `Block` returns.

```scrut
$ wo 'Block[{x = 10}, x^2]'
100
```


## `Catch` / `Throw`

Non-local exit — `Throw[val]` unwinds until the innermost `Catch`.

```scrut
$ wo 'Catch[Throw[42]]'
42
```

```scrut
$ wo 'Catch[Do[If[i == 3, Throw[i]], {i, 1, 5}]]'
3
```


## `Check`

Returns its first argument unless a message is emitted, in which case
it returns the second argument.

```scrut
$ wo 'Check[1/2, error]'
1/2
```


## `Quiet`

Evaluates an expression while suppressing its messages.

```scrut
$ wo 'Quiet[1/0]'
ComplexInfinity
```


## `Return`

Inside a function body, exits early returning a value.

```scrut
$ wo 'f[x_] := Return[x + 1]; f[5]'
6
```


## `DateString`

```scrut
$ wo 'StringStartsQ[DateString[Now, "ISODateTime"], "2026-"]'
True
```
