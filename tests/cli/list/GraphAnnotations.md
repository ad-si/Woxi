# Graph annotations

A graph carries its annotations as options of its own, which `Options` reads
back:

```scrut
$ wo 'Options[Graph[{1 <-> 2, 2 <-> 3}, EdgeWeight -> {5, 7}]]'
{EdgeWeight -> {5, 7}}
```

`AnnotationValue` and `PropertyValue` name one vertex or edge to read the
value it carries; a property the graph does not have gives `$Failed`:

```scrut
$ wo 'PropertyValue[{Graph[{1 <-> 2, 2 <-> 3}, EdgeWeight -> {5, 7}], 2 <-> 3}, EdgeWeight]'
7
```

```scrut
$ wo 'AnnotationValue[{Graph[{1 <-> 2}, EdgeWeight -> {5}], 1}, VertexWeight]'
$Failed
```

`SetProperty` writes one:

```scrut
$ wo 'Options[SetProperty[{Graph[{1 <-> 2}], 1 <-> 2}, EdgeWeight -> 7]]'
{EdgeWeight -> {7}}
```

The weight predicates say what kind of weights a graph carries:

```scrut
$ wo 'EdgeWeightedGraphQ[Graph[{1 <-> 2}, EdgeWeight -> {5}]]'
True
```

```scrut
$ wo 'VertexWeightedGraphQ[Graph[{1 <-> 2}, EdgeWeight -> {5}]]'
False
```

In an `EdgeTaggedGraph` every edge carries a tag; one written without a tag
is tagged by how many copies of it came before:

```scrut
$ wo 'ToString[EdgeList[EdgeTaggedGraph[{1 <-> 2, 2 <-> 3}]], InputForm]'
{UndirectedEdge[1, 2, 1], UndirectedEdge[2, 3, 1]}
```

```scrut
$ wo 'EdgeTags[EdgeTaggedGraph[{UndirectedEdge[1, 2, "a"], 2 <-> 3}]]'
{a, 1}
```
