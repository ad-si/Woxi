# `DecimalForm`

Formats a number in ordinary decimal (non-scientific) notation.

```scrut
$ wo 'ToString[DecimalForm[3.14159]]'
3.14159
```

Large numbers keep their integer part rather than switching to scientific
notation.

```scrut
$ wo 'ToString[DecimalForm[1234567.89]]'
1234568.
```

A second argument gives the number of significant figures.

```scrut
$ wo 'ToString[DecimalForm[1234.5678, 6]]'
1234.57
```
