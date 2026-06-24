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
