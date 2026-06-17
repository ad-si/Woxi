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

A list of strings threads, giving one association per string.

```scrut
$ wo 'CharacterCounts[{"hello", "world"}]'
{<|l -> 2, o -> 1, e -> 1, h -> 1|>, <|d -> 1, l -> 1, r -> 1, o -> 1, w -> 1|>}
```
