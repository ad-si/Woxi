# `CenterArray`

Center a list within a larger array.

```scrut
$ wo 'CenterArray[{a, b, c}, 2]'
{b, c}
```

A scalar centers a single element, padding the rest with `0`:

```scrut
$ wo 'CenterArray[x, 5]'
{0, 0, x, 0, 0}
```

Multi-dimensional specifications produce nested arrays:

```scrut
$ wo 'CenterArray[x, {5, 5}]'
{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, x, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}}
```

With a single argument, the centered element defaults to `1`:

```scrut
$ wo 'CenterArray[5]'
{0, 0, 1, 0, 0}
```
