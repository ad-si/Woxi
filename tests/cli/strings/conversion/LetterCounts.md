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

A list of strings threads, giving one association per string.

```scrut
$ wo 'LetterCounts[{"hello", "world", "!"}]'
{<|l -> 2, o -> 1, e -> 1, h -> 1|>, <|d -> 1, l -> 1, r -> 1, o -> 1, w -> 1|>, <||>}
```
