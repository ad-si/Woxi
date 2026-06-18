# `WeaklyConnectedGraphQ`

Tests whether a directed graph is weakly connected — that is, whether its
underlying undirected graph (edge directions ignored) is connected.

```scrut
$ wo 'WeaklyConnectedGraphQ[Graph[{1 -> 2, 2 -> 3}]]'
True
```

Disconnected pieces are not weakly connected.

```scrut
$ wo 'WeaklyConnectedGraphQ[Graph[{1 -> 2, 3 -> 4}]]'
False
```
