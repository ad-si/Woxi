# `ExampleData`

`ExampleData` serves the datasets bundled with Woxi. `ExampleData[]` lists the
available types.

```scrut
$ wo 'ExampleData[]'
{NetworkGraph}
```

`ExampleData["type"]` lists the entries of that type, each a `{type, name}`
pair ready to be passed straight back in.

```scrut
$ wo 'ExampleData["NetworkGraph"][[1]]'
{NetworkGraph, ZacharyKarateClub}
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

```scrut
$ wo 'ExampleData[{"NetworkGraph", "LesMiserables"}, "VertexList"][[1 ;; 3]]'
{Myriel, Napoleon, MlleBaptistine}
```

A dataset that is not bundled stays unevaluated rather than returning
something made up.

```scrut
$ wo 'ExampleData[{"NetworkGraph", "NoSuchNetwork"}]'
ExampleData[{NetworkGraph, NoSuchNetwork}]
```
