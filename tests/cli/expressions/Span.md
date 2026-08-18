# `Span`

Extract a span of elements using ;; notation in Part.

```scrut
$ wo 'Range[10][[3 ;; 6]]'
{3, 4, 5, 6}
```

Leaving out an end runs the span to the last element, and a third part is
the step:

```scrut
$ wo 'Range[6][[2 ;; ;; 2]]'
{2, 4, 6}
```

`;;` binds tighter than `->`, so a span can be either side of a rule.
`StringExtract` reads such a rule as "split at the delimiter, then take
this span of the parts":

```scrut
$ wo 'StringExtract["a--bbb--ccc--dddd", "--" -> 3 ;;]'
{ccc, dddd}
```

```scrut
$ wo 'Head[1 ;; 2 -> b]'
Rule
```

Every operator Wolfram places below `;;` — assignment, comparison, the
logical operators, `|`, `~~` and `->` — keeps the whole span as its operand,
so a symbol set to a span holds the span itself:

```scrut
$ wo 'x = 2 ;; 4; Range[6][[x]]'
{2, 3, 4}
```

```scrut
$ wo 'a == 1 ;; 3'
a == Span[1, 3]
```

```scrut
$ wo 'a && 1 ;; 3'
a && Span[1, 3]
```

The arithmetic operators bind tighter and stay inside the span's operands:

```scrut
$ wo '1 ;; 2 + 3'
Span[1, 5]
```
