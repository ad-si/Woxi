# `` Developer`ToPackedArray ``

Packs a full rectangular array of machine numbers into a single numeric type.
Only an array whose leaves already share one type can be packed, so packing on
its own leaves every element exactly as it was.

```scrut
$ wo 'Developer`ToPackedArray[{1, 2, 3}]'
{1, 2, 3}
```

An array mixing integers and reals is not packable and comes back unchanged.

```scrut
$ wo 'Developer`ToPackedArray[{1, 2.5}]'
{1, 2.5}
```

A second argument asks for a specific type and converts the leaves to it.

```scrut
$ wo 'Developer`ToPackedArray[{1, 2}, Real]'
{1., 2.}
```

```scrut
$ wo 'Developer`ToPackedArray[{1, 2}, Complex]'
{1. + 0.*I, 2. + 0.*I}
```

Reals whose value is a whole number can be demoted to integers.

```scrut
$ wo 'Developer`ToPackedArray[{1., 2.}, Integer]'
{1, 2}
```

Anything that cannot be packed — ragged nesting, rationals, symbols — is
returned unchanged.

```scrut
$ wo 'Developer`ToPackedArray[{1, {2, 3}}]'
{1, {2, 3}}
```

```scrut
$ wo 'Developer`ToPackedArray[{1/2, 1}]'
{1/2, 1}
```
