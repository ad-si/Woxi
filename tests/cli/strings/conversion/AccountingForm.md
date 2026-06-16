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

A second argument gives the number of significant figures.

```scrut
$ wo 'ToString[AccountingForm[1234.5678, 3]]'
1230.
```
