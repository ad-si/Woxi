# `Dendrogram`

Constructs a dendrogram from the hierarchical clustering of data.

Each element becomes a leaf; clusters are merged bottom-up and every merge
is drawn as an elbow connector at the height of the cluster dissimilarity.

```scrut
$ wo 'Head[Dendrogram[{1, 2, 5, 6, 12}]]'
Graphics
```

Numeric vectors are clustered by Euclidean distance.

```scrut
$ wo 'Head[Dendrogram[{{1, 1}, {1, 2}, {10, 10}, {10, 11}}]]'
Graphics
```

Strings are clustered by edit distance.

```scrut
$ wo 'Head[Dendrogram[{"apple", "aple", "banana", "bananna"}]]'
Graphics
```

Leaves can be labeled with `element -> label` rules, a
`{elements} -> {labels}` rule, or an association keyed by label.

```scrut
$ wo 'Head[Dendrogram[{1 -> "one", 2 -> "two", 10 -> "ten"}]]'
Graphics
```

```scrut
$ wo 'Head[Dendrogram[<|"x" -> 1, "y" -> 2, "z" -> 10|>]]'
Graphics
```

A second argument sets the orientation of the root: `Top` (default),
`Bottom`, `Left` or `Right`.

```scrut
$ wo 'Head[Dendrogram[{1, 2, 5, 6, 12}, Left]]'
Graphics
```

### Options

- **`DistanceFunction`** — how to measure the distance between two data
  elements (e.g. `EuclideanDistance`, `ManhattanDistance` or a pure
  function).
- **`ClusterDissimilarityFunction`** — the linkage used to measure the
  dissimilarity between two clusters: `"Single"`, `"Complete"`,
  `"Average"`, `"WeightedAverage"`, `"Centroid"`, `"Median"` or `"Ward"`.
- **`ImageSize`** — overall size in pixels.

```scrut
$ wo 'Head[Dendrogram[{1, 2, 5, 6, 12}, ClusterDissimilarityFunction -> "Single", DistanceFunction -> ManhattanDistance]]'
Graphics
```
