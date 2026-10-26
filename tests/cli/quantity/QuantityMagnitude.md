# `QuantityMagnitude`

Returns the magnitude of a quantity.

```scrut
$ wo 'QuantityMagnitude[Quantity[5, "Meters"]]'
5
```

A plain number is a dimensionless quantity and is returned unchanged.

```scrut
$ wo 'QuantityMagnitude[0.1]'
0.1
```
