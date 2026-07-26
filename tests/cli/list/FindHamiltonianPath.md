# `FindHamiltonianPath`

`FindHamiltonianPath[g]` gives the vertices of one path that visits every
vertex of `g` exactly once, or `{}` when the graph has none:

```scrut
$ wo '{FindHamiltonianPath[CycleGraph[4]], FindHamiltonianPath[Graph[{1 <-> 2, 3 <-> 4}]]}'
{{1, 2, 3, 4}, {}}
```

Directed edges are followed in their own direction only:

```scrut
$ wo 'FindHamiltonianPath[Graph[{1 -> 2, 2 -> 3}]]'
{1, 2, 3}
```

`FindHamiltonianPath[g, s, t]` asks for a path from `s` to `t`:

```scrut
$ wo '{FindHamiltonianPath[Graph[{1 <-> 2, 2 <-> 3}], 3, 1], FindHamiltonianPath[Graph[{1 <-> 2, 2 <-> 3}], 2, 3]}'
{{3, 2, 1}, {}}
```
