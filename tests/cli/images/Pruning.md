# `Pruning`

Removes the outermost branches of thin objects in an image.

`Pruning[image, n]` runs `n` passes, each of which deletes every *endpoint* —
a foreground pixel at the tip of a branch. Here that is the two ends of the
bar and the pixel sticking out of its middle:

```scrut
$ wo 'ImageData[Pruning[Image[{{0, 0, 1, 0, 0}, {1, 1, 1, 1, 1}}], 1]]'
{{0., 0., 0., 0., 0.}, {0., 1., 1., 1., 0.}}
```

Every pass eats one more pixel off each branch, so `n` passes remove every
branch at most `n` pixels long:

```scrut
$ wo 'ImageData[Pruning[Image[{{0, 0, 1, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 1, 0, 0}, {1, 1, 1, 1, 1}}], 2]]'
{{0., 0., 0., 0., 0.}, {0., 0., 0., 0., 0.}, {0., 0., 1., 0., 0.}, {0., 0., 1., 0., 0.}}
```

`Infinity` keeps going until nothing more falls away:

```scrut
$ wo 'ImageData[Pruning[Image[{{0, 0, 1, 0, 0}, {0, 0, 1, 0, 0}, {1, 1, 1, 1, 1}}], Infinity]]'
{{0., 0., 0., 0., 0.}, {0., 0., 0., 0., 0.}, {0., 0., 1., 0., 0.}}
```

A connected piece that never branches is a shape rather than a branch, so
`Pruning[image, n]` leaves it whole however many passes are asked for:

```scrut
$ wo 'ImageData[Pruning[Image[{{1, 1, 1, 1, 1}}], Infinity]]'
{{1., 1., 1., 1., 1.}}
```

A pixel with no lit neighbor at all is an isolated point, not the tip of a
branch, and survives:

```scrut
$ wo 'ImageData[Pruning[Image[{{1, 0}, {0, 0}}]]]'
{{1., 0.}, {0., 0.}}
```

Only the pruned pixels go black; the survivors keep their gray value:

```scrut
$ wo 'ImageData[Pruning[Image[{{0.4, 0.6, 0.8}}]]]'
{{0., 0.6000000238418579, 0.}}
```

A branch length that is not a non-negative integer is reported:

```scrut
$ wo 'Pruning[Image[{{1, 0}}], -1]'

Pruning::invniter: Expecting a non-negative integer value for the number of iterations.
Pruning[-Image-, -1]
```
