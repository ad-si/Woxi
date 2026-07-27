# `NumberForm`

Prints a number to a given precision. `NumberForm[x, n]` shows `n`
significant figures, `NumberForm[x, {n, f}]` shows `n` digits with `f` of
them after the decimal point.

```scrut
$ wo 'ToString[NumberForm[1234.5678, 6]]'
1234.57
```

```scrut
$ wo 'ToString[NumberForm[50., {3, 1}]]'
50.0
```

`NumberPoint` replaces the decimal point:

```scrut
$ wo 'ToString[NumberForm[1234.5, NumberPoint -> ","]]'
1234,5
```

`DigitBlock` groups the digits — from the right on the integer side and from
the left on the fractional side — with `NumberSeparator` between the blocks:

```scrut
$ wo 'ToString[NumberForm[1234.5678, {8, 4}, DigitBlock -> 2]]'
12,34.56 78
```

```scrut
$ wo 'ToString[NumberForm[1234.5678, {8, 4}, DigitBlock -> 3, NumberSeparator -> {"|", "_"}]]'
1|234.567_8
```

`NumberPadding -> {left, right}` fills the field: the left string pads the
integer columns, the right one the fractional slots past the value's
significant digits.

```scrut
$ wo 'ToString[NumberForm[12.3, {8, 3}, NumberPadding -> {"*", "0"}]]'
****12.300
```

By default the sign stays next to the digits; `SignPadding -> True` moves it
in front of the padding:

```scrut
$ wo 'ToString[NumberForm[-12.3, {6, 2}, NumberPadding -> {"0", "0"}]]'
00-12.30
```

```scrut
$ wo 'ToString[NumberForm[-12.3, {6, 2}, SignPadding -> True, NumberPadding -> {"0", "0"}]]'
-0012.30
```

An argument that is not an approximate number renders as itself:

```scrut
$ wo 'ToString[NumberForm[Pi, 5]]'
Pi
```
