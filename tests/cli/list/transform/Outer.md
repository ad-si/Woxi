# `Outer`

Generalized outer product - applies function to all pairs.

```scrut
$ wo 'Outer[Times, {1, 2}, {3, 4}]'
{{3, 4}, {6, 8}}
```

```scrut
$ wo 'Outer[Plus, {1, 2}, {10, 20}]'
{{11, 21}, {12, 22}}
```

A single argument to range over makes it a `Map`, and every argument must be
nonatomic and share one head:

```scrut
$ wo 'Outer[f, {1, 2}]'
{f[1], f[2]}
```

```scrut
$ wo 'Outer[f, {1, 2}, h[3, 4]]'

Outer::heads: Heads h and List at positions 3 and 2 are expected to be the same.
Outer[f, {1, 2}, h[3, 4]]
```
