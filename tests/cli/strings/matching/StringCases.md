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

The `Overlaps` option controls how much of the string a match may share with
its neighbours. The default keeps matches disjoint, `Overlaps -> True` reports
the preferred match at every start position, and `Overlaps -> All` reports
*every* match at every start position:

```scrut
$ wo 'StringCases["abcd", __]'
{abcd}
```

```scrut
$ wo 'StringCases["abcd", __, Overlaps -> True]'
{abcd, bcd, cd, d}
```

```scrut
$ wo 'StringCases["abcd", __, Overlaps -> All]'
{abcd, abc, ab, a, bcd, bc, b, cd, c, d}
```

A greedy pattern reports its longest match first at each start position, so
`Shortest` flips the order within each group:

```scrut
$ wo 'StringCases["abcd", Shortest[__], Overlaps -> All]'
{a, ab, abc, abcd, b, bc, bcd, c, cd, d}
```

A `DatePattern` matches the date fields it names, each read whole:

```scrut
$ wo 'StringCases["2024-01-15", DatePattern[{"Year", "Month", "Day"}]]'
{2024-01-15}
```

```scrut
$ wo 'StringCases["31/12/1999", DatePattern[{"Day", "Month", "Year"}]]'
{31/12/1999}
```

A field out of range matches nothing:

```scrut
$ wo 'StringCases["2024-13-01", DatePattern[{"Year", "Month", "Day"}]]'
{}
```
