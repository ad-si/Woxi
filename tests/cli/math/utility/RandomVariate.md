# `RandomVariate`

Generates pseudorandom variates from a distribution.

```scrut
$ wo 'Length[RandomVariate[NormalDistribution[0, 1], 50]]'
50
```

A uniform range with a single point in it yields that point:

```scrut
$ wo 'RandomVariate[UniformDistribution[{1, 1}], 3]'
{1., 1., 1.}
```

Bounds given the wrong way round name the same interval:

```scrut
$ wo 'AllTrue[RandomVariate[UniformDistribution[{7, 3}], 100], (# >= 3 && # < 7) &]'
True
```
