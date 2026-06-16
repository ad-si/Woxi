# `LetterCounts`

Returns an association mapping each letter to its number of occurrences.

```scrut
$ wo 'LetterCounts["banana"]'
<|a -> 3, n -> 2, b -> 1|>
```

With a second argument it counts letter n-grams; non-letters break the window.

```scrut
$ wo 'LetterCounts["ab12cd34ab12", 2]'
<|ab -> 2, cd -> 1|>
```
