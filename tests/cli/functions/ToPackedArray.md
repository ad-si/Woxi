# `` Developer`ToPackedArray ``

Packs a full rectangular array of machine numbers into a single numeric type.
Woxi keeps no packed representation, so the only visible effect is that type
unification: an array mixing integers and reals comes back as all reals.

```scrut
$ wo 'Developer`ToPackedArray[{1, 2, 3}]'
{1, 2, 3}
```

```scrut
$ wo 'Developer`ToPackedArray[{1, 2.5}]'
{1., 2.5}
```

A second argument asks for a specific type.

```scrut
$ wo 'Developer`ToPackedArray[{1, 2}, Real]'
{1., 2.}
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
