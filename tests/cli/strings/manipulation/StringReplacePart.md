# `StringReplacePart`

Replaces a specific character range (1-indexed, inclusive) with a new string.

```scrut
$ wo 'StringReplacePart["Hello world", "XXXX", {1, 5}]'
XXXX world
```

A bare index stands for the span from the start, so replacing at `2` replaces
the first two characters:

```scrut
$ wo 'StringReplacePart["abc", "x", 2]'
xc
```

A span the string does not have is reported and the call left alone:

```scrut
$ wo 'StringReplacePart["abc", "x", {5, 6}]'

StringReplacePart::repart: Cannot replace positions 5 through 6 in "abc".
StringReplacePart[abc, x, {5, 6}]
```
