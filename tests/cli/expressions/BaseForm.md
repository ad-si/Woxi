# `BaseForm`

Displays a number in a specified base. Without a front-end it prints verbatim,
matching `wolframscript`.

```scrut
$ wo 'BaseForm[255, 16]'
BaseForm[255, 16]
```

Under `ToString` the digits are rendered with the base shown as a subscript on
the line below.

```scrut
$ wo 'ToString[BaseForm[255, 16]]'
ff
  16
```

Base 10 shows no subscript.

```scrut
$ wo 'ToString[BaseForm[255, 10]]'
255
```
