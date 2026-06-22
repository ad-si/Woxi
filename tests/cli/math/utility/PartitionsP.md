# `PartitionsP`

Number of unrestricted partitions of an integer.

```scrut
$ wo 'PartitionsP[0]'
1
```

```scrut
$ wo 'PartitionsP[10]'
42
```

Negative integers have no partitions:

```scrut
$ wo 'PartitionsP[-3]'
0
```

`PartitionsP` is `Listable` and threads over lists:

```scrut
$ wo 'PartitionsP[{4, 5, 6}]'
{5, 7, 11}
```

It stays unevaluated for non-integer arguments:

```scrut
$ wo 'PartitionsP[3.5]'
PartitionsP[3.5]
```
