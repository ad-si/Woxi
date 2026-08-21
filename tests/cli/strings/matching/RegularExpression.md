# `RegularExpression`

Represents a regular expression pattern for string matching.

```scrut
$ wo 'StringCases["a1b22c", RegularExpression["[0-9]+"]]'
{1, 22}
```

```scrut
$ wo 'StringMatchQ["2024-05-01", RegularExpression["\\d{4}-\\d{2}-\\d{2}"]]'
True
```

```scrut
$ wo 'StringReplace["one two", RegularExpression["\\s+"] -> "-"]'
one-two
```

## Escaping

The syntax is PCRE, where a backslash before a character that needs no
escaping simply means that character. Patterns written by hand often escape
liberally, and those escapes are accepted:

```scrut
$ wo 'StringReplace["a<b", RegularExpression["\\<"] -> "-"]'
a-b
```

```scrut
$ wo 'StringMatchQ["50%", RegularExpression["\\d+\\%"]]'
True
```

An escape that does mean something keeps its meaning — `\.` still matches
only a literal dot:

```scrut
$ wo 'StringCases["a.b axb", RegularExpression["a\\.b"]]'
{a.b}
```
