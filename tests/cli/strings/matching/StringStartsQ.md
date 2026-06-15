# `StringStartsQ`

```scrut
$ wo 'StringStartsQ["Hello World!", "Hello"]'
True
```

```scrut
$ wo 'StringStartsQ["Hello World!", "Bye"]'
False
```

The one-argument operator form applies to a string later, e.g. inside `Select`.

```scrut
$ wo 'Select[{"CAC1", "CTG1", "ACT1", "CGA1", "CTC1"}, StringStartsQ["C"]]'
{CAC1, CTG1, CGA1, CTC1}
```
