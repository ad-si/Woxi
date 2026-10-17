# `Dot`

```scrut
$ wo 'Dot[{1, 2, 3}, {4, 5, 6}]'
32
```

```scrut
$ wo 'Dot[{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}]'
{{19, 22}, {43, 50}}
```


`Dot` is `Flat`, so nested products are spliced in
and an empty `Dot[]` disappears.

```scrut
$ wo 'Dot[a, Dot[b, c], Dot[], d]'
a . b . c . d
```

```scrut
$ wo 'Attributes[Dot]'
{Flat, OneIdentity, Protected}
```
