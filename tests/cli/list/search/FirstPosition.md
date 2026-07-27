# `FirstPosition`

Returns the position of the first occurrence of an expression.

```scrut
$ wo 'FirstPosition[{1, 2, 3, 2, 1}, 2]'
{2}
```

It reports the first position `Position` finds, heads included — the head of
an expression sits at position `{0}`, so it is reached before any element:

```scrut
$ wo 'FirstPosition[{1, a, 2}, _Symbol]'
{0}
```

```scrut
$ wo 'FirstPosition[{1, 2}, _Integer]'
{1}
```

The whole expression is position `{}`, and an association element is located
by its key:

```scrut
$ wo 'FirstPosition[{1, a}, _List]'
{}
```

```scrut
$ wo 'ToString[FirstPosition[<|"a" -> 1, "b" -> 2|>, 2], InputForm]'
{Key["b"]}
```

Nothing found gives the default:

```scrut
$ wo 'FirstPosition[{1, 2, 3}, 5]'
Missing[NotFound]
```

```scrut
$ wo 'FirstPosition[{1, 2, 3}, 5, none]'
none
```
