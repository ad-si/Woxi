# `StringEndsQ`

```scrut
$ wo 'StringEndsQ["Hello World!", "World!"]'
True
```

```scrut
$ wo 'StringEndsQ["Hello World!", "Moon!"]'
False
```

The one-argument operator form applies to a string later, e.g. inside `Select`.

```scrut
$ wo 'Select[{"CAC1", "CTG1", "ACT1", "CGA1", "CTC1"}, StringEndsQ["C1"]]'
{CAC1, CTC1}
```
