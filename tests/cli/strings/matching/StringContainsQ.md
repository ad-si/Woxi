# `StringContainsQ`

Tests whether a substring occurs in a string.

```scrut
$ wo 'StringContainsQ["Hello world", "world"]'
True
```

```scrut
$ wo 'StringContainsQ["Hello world", "planet"]'
False
```

The one-argument operator form applies to a string later, e.g. inside `Select`.

```scrut
$ wo 'Select[{"CAC1", "CTG1", "ACT1", "CGA1", "CTC1"}, StringContainsQ["G"]]'
{CTG1, CGA1}
```
