# `StringPadLeft`

Pads a string on the left to a specified length.

```scrut
$ wo 'StringPadLeft["hi", 5]'
   hi
```

```scrut
$ wo 'StringPadLeft["hi", 5, "0"]'
000hi
```

```scrut
$ wo 'StringPadLeft["hello", 3]'
llo
```

Given just a list of strings, each is padded to the length of the longest.

```scrut
$ wo 'StringPadLeft[{"a", "ab", "abc"}]'
{  a,  ab, abc}
```
