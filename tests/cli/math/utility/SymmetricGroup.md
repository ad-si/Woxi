# `SymmetricGroup`

represents the symmetric group `S_n` of degree `n`.
It stays unevaluated as its canonical form and is consumed by
group functions such as `GroupOrder` and `GroupGenerators`.

```scrut
$ wo 'SymmetricGroup[3]'
SymmetricGroup[3]
```

```scrut
$ wo 'GroupOrder[SymmetricGroup[4]]'
24
```

```scrut
$ wo 'GroupGenerators[SymmetricGroup[4]]'
{Cycles[{{1, 2}}], Cycles[{{1, 2, 3, 4}}]}
```
