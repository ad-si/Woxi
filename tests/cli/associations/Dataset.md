# `Dataset`

Wraps data with type information for structured data handling.

A dataset is queried with the same successive-level operator spec `Query` takes.
An operator that yields a collection gives back a dataset,
while one that yields an atom returns it bare.

```scrut
$ wo 'Normal[Dataset[{<|"a" -> 1|>, <|"a" -> 2|>, <|"a" -> 3|>}][Select[#a > 1 &]]]'
{<|a -> 2|>, <|a -> 3|>}
```

```scrut
$ wo 'Dataset[{<|"a" -> 1|>, <|"a" -> 2|>}][Total, "a"]'
3
```

An integer selects a row:

```scrut
$ wo 'Normal[Dataset[{<|"a" -> 1|>, <|"a" -> 2|>}][2]]'
<|a -> 2|>
```
