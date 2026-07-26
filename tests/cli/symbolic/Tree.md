# `Tree`

`Tree[data, {children…}]` is a tree node; `None` in place of the children
makes a leaf. The data and the children are read back with `TreeData` and
`TreeChildren`:

```scrut
$ wo 't = Tree[1, {Tree[2, None], Tree[3, None]}]; {TreeData[t], Length[TreeChildren[t]]}'
{1, 2}
```

`TreeLeaves` gives the leaf subtrees, left to right:

```scrut
$ wo 'TreeLeaves[Tree[1, {Tree[2, {Tree[4, None]}], Tree[3, None]}]]'
{Tree[4, None], Tree[3, None]}
```

`TreeCases` gives the subtrees whose data matches, children before their
parent, and `TreeScan` visits them in that same order:

```scrut
$ wo 'TreeCases[Tree[1, {Tree[2, {Tree[4, None]}], Tree[3, None]}], _?EvenQ]'
{Tree[4, None], Tree[2, {Tree[4, None]}]}
```

```scrut
$ wo 'Reap[TreeScan[Sow, Tree[1, {Tree[2, {Tree[4, None]}], Tree[3, None]}]]]'
{Null, {{4, 2, 3, 1}}}
```

A tree is an atom: it has no parts, no depth beyond itself, and counts as a
single leaf — so `Map` and `Apply` leave it alone:

```scrut
$ wo '{AtomQ[Tree[1, None]], Depth[Tree[1, {Tree[2, None]}]], Length[Tree[1, None]], LeafCount[Tree[1, {Tree[2, None]}]]}'
{True, 1, 0, 1}
```

```scrut
$ wo 'Map[f, Tree[1, {Tree[2, None]}]]'
Tree[1, {Tree[2, None]}]
```
