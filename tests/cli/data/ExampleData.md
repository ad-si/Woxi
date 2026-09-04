# `ExampleData`

`ExampleData` serves the datasets bundled with Woxi. `ExampleData[]` lists the
available types.

```scrut
$ wo 'MemberQ[ExampleData[], "NetworkGraph"]'
True
```

`ExampleData["type"]` lists the entries of that type, each a `{type, name}`
pair ready to be passed straight back in.

```scrut
$ wo 'ExampleData["NetworkGraph"][[1]]'
{NetworkGraph, AmericanCollegeFootball}
```

`ExampleData[{"type", "name"}]` is the data itself — for a network, a `Graph`.

```scrut
$ wo 'EdgeCount[ExampleData[{"NetworkGraph", "ZacharyKarateClub"}]]'
78
```

A second argument selects one property instead.

```scrut
$ wo 'ExampleData[{"NetworkGraph", "LesMiserables"}, "VertexCount"]'
77
```

The data itself is a real `Graph`, so the graph functions work on it:

```scrut
$ wo 'MemberQ[VertexList[ExampleData[{"NetworkGraph", "LesMiserables"}]], "Napoleon"]'
True
```

A name that is not in the catalogue at all is reported.

```scrut
$ wo 'ExampleData[{"NetworkGraph", "NoSuchNetwork"}]'

ExampleData::notent: "NoSuchNetwork" is not a known entity for the collection "NetworkGraph". Use ExampleData["NetworkGraph"] for a list of entities.
ExampleData[{NetworkGraph, NoSuchNetwork}]
```

A catalogued dataset Woxi does not bundle stays unevaluated rather than
returning something made up.
