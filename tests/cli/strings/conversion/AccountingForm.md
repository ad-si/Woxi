# `AccountingForm`

Formats a number in accounting notation: like `NumberForm`, but negative
numbers are shown in parentheses instead of with a minus sign.

```scrut
$ wo 'ToString[AccountingForm[1234.5]]'
1234.5
```

```scrut
$ wo 'ToString[AccountingForm[-1234.5]]'
(1234.5)
```

A second argument gives the number of significant figures. Requesting fewer
figures than the number has integer digits pads the trailing digits with zeros
and emits the `reqsigz` warning.

```scrut {output_stream: combined}
$ wo 'ToString[AccountingForm[1234.5678, 3]]'

AccountingForm::reqsigz: Requested number precision is lower than number of digits shown; padding with zeros.
1230.
```

A `{n, f}` spec fixes the digits after the decimal point. The closing bracket
takes a column of its own, so a padded fraction is one slot wider than `f`:

```scrut
$ wo 'ToString[AccountingForm[-12.3, {5, 2}]]'
(12.3)
```

```scrut
$ wo 'ToString[AccountingForm[12.3, {6, 3}, NumberPadding -> {" ", "0"}]]'
  12.3000
```

`NumberSigns` replaces the brackets:

```scrut
$ wo 'ToString[AccountingForm[-12.3, NumberSigns -> {"<", ">"}]]'
<12.3
```

A list is formatted element-wise:

```scrut
$ wo 'ToString[AccountingForm[{-1.5, 2.5}]]'
{(1.5), 2.5}
```
