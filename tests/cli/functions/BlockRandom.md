# `BlockRandom`

Evaluates an expression with the random generator state localized, so a
`SeedRandom` inside the block does not shift the outer random sequence.
Only the body is held:

```scrut
$ wo 'Attributes[BlockRandom]'
{HoldFirst, Protected}
```

The block returns its body's value:

```scrut
$ wo 'BlockRandom[1 + 1]'
2
```

The draw after the block is the one that would have come next anyway, even
though the block reseeded:

```scrut
$ wo 'SeedRandom[42]; a = RandomInteger[1000]; BlockRandom[SeedRandom[1]; RandomInteger[1000]]; b = RandomInteger[1000]; SeedRandom[42]; RandomInteger[1000]; a != b && b == RandomInteger[1000]'
True
```

`RandomSeeding -> seed` seeds the localized generator, making the block
reproducible without disturbing the ambient state:

```scrut
$ wo 'BlockRandom[RandomInteger[10^6], RandomSeeding -> 42] == BlockRandom[RandomInteger[10^6], RandomSeeding -> 42]'
True
```

`RandomSeeding -> Inherited` is the default — the block sees the ambient
state:

```scrut
$ wo 'SeedRandom[7]; BlockRandom[RandomInteger[10^6], RandomSeeding -> Inherited] == RandomInteger[10^6]'
True
```
