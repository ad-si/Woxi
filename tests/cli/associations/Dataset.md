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

Statistics and list functions applied to a dataset see the data it wraps,
not its internal `Dataset[data, type, metadata]` structure:

```scrut
$ wo 'Total[Dataset[{1, 2, 3}]]'
6
```

```scrut
$ wo 'Mean[Dataset[{1, 2, 3}]]'
2
```

A function that reduces to a single value returns it bare,
while one that yields a collection gives back a dataset:

```scrut
$ wo 'Head[Total[Dataset[{1, 2}]]]'
Integer
```

```scrut
$ wo 'Head[Sort[Dataset[{3, 1, 2}]]]'
Dataset
```

```scrut
$ wo 'Normal[Sort[Dataset[{3, 1, 2}]]]'
{1, 2, 3}
```

`Part` indexes the data as well:

```scrut
$ wo 'Dataset[{{1, 2}, {3, 4}}][[2, 1]]'
3
```

```scrut
$ wo 'Dataset[{<|"a" -> 1, "b" -> 2|>}][[1, "a"]]'
1
```

A string in a grouping or sorting operator extracts that key:

```scrut
$ wo 'Normal[Dataset[{<|"a" -> 1, "b" -> 5|>, <|"a" -> 1, "b" -> 6|>}][GroupBy["a"]]]'
<|1 -> {<|a -> 1, b -> 5|>, <|a -> 1, b -> 6|>}|>
```

The operator form of `Query` applies the same spec to a dataset:

```scrut
$ wo 'Query[Total][Dataset[{1, 2, 3}]]'
6
```

```scrut
$ wo 'Normal[Query[All, "a"][Dataset[{<|"a" -> 1|>, <|"a" -> 2|>}]]]'
{1, 2}
```

A bare rule at an operator position is read as an option to `Query`,
so the level is left untouched:

```scrut {output_stream: combined}
$ wo 'Normal[Dataset[{<|"a" -> 1|>, <|"a" -> 2|>}][All, "a" -> "b"]]'

OptionValue::nodef: Unknown option a for Query.
{<|a -> 1|>, <|a -> 2|>}
```

A dataset is an atom for traversal, while `Length` and `Dimensions`
still describe the data it wraps:

```scrut
$ wo 'AtomQ[Dataset[{1, 2}]]'
True
```

```scrut
$ wo 'Depth[Dataset[{1, 2}]]'
1
```

```scrut
$ wo 'Length[Dataset[{1, 2}]]'
2
```

```scrut
$ wo 'Dimensions[Dataset[{{1, 2}, {3, 4}}]]'
{2, 2}
```
