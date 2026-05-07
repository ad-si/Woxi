# `StringCases`

Finds all occurrences of a substring.

```scrut
$ wo 'StringCases["abcabc", "bc"]'
{bc, bc}
```

```scrut
$ wo 'StringCases["hello", "l"]'
{l, l}
```

```scrut
$ wo 'StringCases["hello", "x"]'
{}
```
