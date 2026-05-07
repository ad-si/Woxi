# `SequenceAlignment`

Aligns two strings, returning a list where common substrings appear
as strings and differences appear as length-2 lists of alternatives.

```scrut
$ wo 'SequenceAlignment["hello", "helloo"]'
{hell, {, o}, o}
```
