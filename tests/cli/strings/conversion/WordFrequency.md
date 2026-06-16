# `WordFrequency`

Gives the fraction of words in a string that equal a given word. Matching is
case-sensitive by default.

```scrut
$ wo 'WordFrequency["a b a c", "a"]'
0.5
```

A list of words gives an association of fractions.

```scrut
$ wo 'WordFrequency["a b a c", {"a", "c"}]'
<|a -> 0.5, c -> 0.25|>
```
