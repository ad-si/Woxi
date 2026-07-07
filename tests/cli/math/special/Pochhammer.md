# `Pochhammer`

Returns the Pochhammer symbol (rising factorial).

```scrut
$ wo 'Pochhammer[3, 0]'
1
```

It threads over either argument.

```scrut
$ wo 'Pochhammer[3, {1, 2}]'
{3, 12}
```

A non-integer rational second argument reduces to the closed form
`Gamma[a + b]/Gamma[a]`.

```scrut
$ wo 'Pochhammer[2, 1/2]'
(3*Sqrt[Pi])/4
```
