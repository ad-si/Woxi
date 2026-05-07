# `ConstantArray`

Creates an array of repeated elements.

```scrut
$ wo 'ConstantArray[x, 3]'
{x, x, x}
```

```scrut
$ wo 'ConstantArray[0, 5]'
{0, 0, 0, 0, 0}
```

```scrut
$ wo 'ConstantArray[1, 0]'
{}
```

```scrut
$ wo 'ConstantArray[a, {2, 3}]'
{{a, a, a}, {a, a, a}}
```
