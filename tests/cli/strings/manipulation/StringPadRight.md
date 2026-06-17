# `StringPadRight`

Pads a string on the right to a specified length.

```scrut
$ wo 'StringPadRight["hi", 5, "0"]'
hi000
```

```scrut
$ wo 'StringPadRight["hello", 3]'
hel
```

Given just a list of strings, each is padded to the length of the longest.

```scrut
$ wo 'StringPadRight[{"a", "ab", "abc"}]'
{a  , ab , abc}
```
