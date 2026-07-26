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

A list of strings is handled string by string, so the matches of each stay
together:

```scrut
$ wo 'StringCases[{"aba", "cd"}, "a"]'
{{a, a}, {}}
```

A subject that is not a string is refused rather than coerced to its printed
form:

```scrut
$ wo 'StringCases[foo, "a"]'

StringCases::strse: A string or list of strings is expected at position 1 in StringCases[foo, a].
StringCases[foo, a]
```
