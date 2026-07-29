# `AddTo` (`+=`)

Increments a variable in place.

```scrut
$ wo 'x = 5; x += 3; x'
8
```

The whole in-place family — `AddTo`, `SubtractFrom`, `TimesBy`, `DivideBy`,
`Increment`, `Decrement`, `AppendTo`, `PrependTo` — needs a target that
already has a value. One that does not is reported and left alone:

```scrut
$ wo 'AddTo[5, 1]'

AddTo::rvalue: 5 is not a variable with a value, so its value cannot be changed.
5 += 1
```

```scrut
$ wo 'AppendTo[{1, 2}, 3]'

AppendTo::rvalue: {1, 2} is not a variable with a value, so its value cannot be changed.
AppendTo[{1, 2}, 3]
```

That includes a symbol that has simply never been set:

```scrut
$ wo 'AppendTo[q, 3]'

AppendTo::rvalue: q is not a variable with a value, so its value cannot be changed.
AppendTo[q, 3]
```

A `Part` target works when the location holds something extendable, and is
reported against the part expression as written when it does not:

```scrut
$ wo 'k = {{1}, {2}}; AppendTo[k[[1]], 9]'
{1, 9}
```

```scrut
$ wo 'm = {1, 2}; AppendTo[m[[1]], 9]'

AppendTo::normal: Nonatomic expression expected at position 1 in AppendTo[m[[1]], 9].
AppendTo[m[[1]], 9]
```

A literal operand parses; it is only at evaluation that there turns out to be
nothing to modify:

```scrut
$ wo '5 += 1'

AddTo::rvalue: 5 is not a variable with a value, so its value cannot be changed.
5 += 1
```

A postfix `++` binds to its operand before juxtaposition, so `2 a++` is
`2 (a++)`:

```scrut
$ wo 'a = 5; 2 a++'
10
```

```scrut
$ wo 'a = 5; a++ 2'
10
```

Because `++` is a single atomic token, `1++2` is `Increment[1] 2` rather than
`1 + (+2)`; separating the signs keeps it arithmetic:

```scrut
$ wo '1++2'

Increment::rvalue: 1 is not a variable with a value, so its value cannot be changed.
2*1++
```

```scrut
$ wo '1 + +2'
3
```
