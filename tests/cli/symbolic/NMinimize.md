# `NMinimize`

Numerical global minimization with constraints using sampling and gradient refinement.

```scrut
$ wo 'NMinimize[(x - 1)^2, x]'
{0., {x -> 1.}}
```

An indexed variable like `n[1]` works too, just like in `Minimize`:

```scrut
$ wo 'NMinimize[{n[1]^2 + n[2]^2, n[1] + n[2] == 1}, {n[1], n[2]}]'
{0.5, {n[1] -> 0.5, n[2] -> 0.5}}
```
