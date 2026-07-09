# `LaguerreL`

Laguerre polynomial.

```scrut
$ wo 'LaguerreL[2, x]'
(2 - 4*x + x^2)/2
```

Substituting a value into the symbolic polynomial reduces to the same number
as evaluating it directly (issue #215):

```scrut
$ wo 'LaguerreL[3, x] /. x -> 3'
1
```
