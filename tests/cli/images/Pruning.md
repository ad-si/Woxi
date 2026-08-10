# `Pruning`

Removes the outermost branches of thin objects in an image.

One pass deletes every *endpoint* — a foreground pixel with exactly one of its
eight neighbors lit. Here that is the two ends of a bar; the pixel sticking out
of its middle has three neighbors below it, so it is a junction rather than a
tip:

```scrut
$ wo 'ImageData[Pruning[Image[{{0, 0, 1, 0, 0}, {1, 1, 1, 1, 1}}]]]'
{{0., 0., 1., 0., 0.}, {0., 1., 1., 1., 0.}}
```

`Pruning[image, n]` repeats the pass `n` times, so it eats back every branch at
most `n` pixels long — one pixel of the spur below per pass:

```scrut
$ wo 'ImageData[Pruning[Image[{{0, 0, 1, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 1, 0, 0}, {1, 1, 1, 1, 1}}], 2]]'
{{0., 0., 0., 0., 0.}, {0., 0., 0., 0., 0.}, {0., 0., 1., 0., 0.}, {0., 1., 1., 1., 0.}}
```

`Infinity` keeps going until nothing more falls away:

```scrut
$ wo 'ImageData[Pruning[Image[{{0, 0, 1, 0, 0}, {0, 0, 1, 0, 0}, {1, 1, 1, 1, 1}}], Infinity]]'
{{0., 0., 0., 0., 0.}, {0., 0., 1., 0., 0.}, {0., 1., 1., 1., 0.}}
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

Pruning::intnm: Non-negative machine-sized integer expected at position 2 in Pruning[-Image-, -1].
Pruning[-Image-, -1]
```
