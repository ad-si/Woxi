# `QPochhammer`

q-Pochhammer symbol.

```scrut
$ wo 'QPochhammer[a, q, 0]'
1
```

The one-argument form is the Euler function `QPochhammer[q] = QPochhammer[q, q]`.

```scrut
$ wo 'QPochhammer[q]'
QPochhammer[q, q]
```

The two-argument form is the infinite product `Product[1 - a q^k, {k, 0, Inf}]`,
evaluated numerically when an argument is inexact.

```scrut
$ wo 'QPochhammer[0.5, 0.5]'
0.2887880950866024
```
