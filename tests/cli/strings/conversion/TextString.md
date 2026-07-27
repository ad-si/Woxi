# `TextString`

Converts an expression to a plain text string (similar to `ToString` but
aimed at human-readable output).

```scrut
$ wo 'TextString[3.14]'
3.14
```

An approximate real is always written in full decimal — never the `*^`
notation `ToString` switches to — and keeps a fractional digit while its
integer part is shorter than six digits:

```scrut
$ wo 'TextString[1234567.]'
1234567.
```

```scrut
$ wo 'TextString[2.]'
2.0
```

An exact number that is not a bare symbol is numericized to six significant
digits, while a symbol prints as itself:

```scrut
$ wo 'TextString[1/3]'
0.333333
```

```scrut
$ wo 'TextString[Sqrt[2]]'
1.41421
```

```scrut
$ wo 'TextString[Pi]'
Pi
```

The rules apply element-wise inside a list or association:

```scrut
$ wo 'TextString[{1/2, Pi}]'
{0.5, Pi}
```

A complex number shows both parts, the imaginary one always as a real:

```scrut
$ wo 'TextString[2 + 3 I]'
2 + 3.0i
```

A missing value contributes nothing to the text:

```scrut
$ wo 'TextString[{Missing["x"], 1}]'
{, 1}
```
