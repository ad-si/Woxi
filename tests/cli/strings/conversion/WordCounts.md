# `WordCounts`

Returns an association mapping each word to its number of occurrences.

```scrut
$ wo 'WordCounts["the cat and the dog"]'
<|the -> 2, dog -> 1, and -> 1, cat -> 1|>
```

Surrounding punctuation is stripped, so `fish,` and `fish` are the same word.

```scrut
$ wo 'WordCounts["one fish, two fish, red fish"]'
<|fish -> 3, red -> 1, two -> 1, one -> 1|>
```

A second argument counts word n-grams.

```scrut
$ wo 'WordCounts["a b a b a", 2]'
<|{b, a} -> 2, {a, b} -> 2|>
```
