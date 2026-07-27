# `PaddedForm`

Right-aligns a number in a field of `n` digit columns plus a column for the
sign. A `{n, f}` spec puts `f` of the digits after the decimal point.

```scrut
$ wo 'ToString[PaddedForm[123, 6]]'
    123
```

```scrut
$ wo 'ToString[PaddedForm[1.5, {4, 2}]]'
  1.50
```

A list pads every element to the same width:

```scrut
$ wo 'ToString[PaddedForm[{1, 22, 333}, 4]]'
{    1,    22,   333}
```

`NumberPadding` given as a single string sets the fill of the integer field:

```scrut
$ wo 'ToString[PaddedForm[-123, 6, NumberPadding -> "0"]]'
000-123
```

```scrut
$ wo 'ToString[PaddedForm[-123, 6, NumberPadding -> "0", SignPadding -> True]]'
-000123
```

`NumberSigns` fills the sign column:

```scrut
$ wo 'ToString[PaddedForm[123, 6, NumberSigns -> {"", "+"}]]'
   +123
```

`DigitBlock` widens the field by the separators a full field would carry:

```scrut
$ wo 'ToString[PaddedForm[123456, 8, DigitBlock -> 3]]'
    123,456
```

Requesting fewer significant figures than the number has integer digits pads
them with zeros and warns:

```scrut {output_stream: combined}
$ wo 'ToString[PaddedForm[1234.5678, 3]]'

PaddedForm::reqsigz: Requested number precision is lower than number of digits shown; padding with zeros.
 1230.
```
