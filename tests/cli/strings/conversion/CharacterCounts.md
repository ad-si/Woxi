# `CharacterCounts`

Returns an association mapping each character to its count.

```scrut
$ wo 'CharacterCounts["banana"]'
<|a -> 3, n -> 2, b -> 1|>
```

With a second argument it counts character n-grams (in first-occurrence order).

```scrut
$ wo 'CharacterCounts["ababcab", 2]'
<|ab -> 3, ba -> 1, bc -> 1, ca -> 1|>
```
