# `UnitConvert`

Converts a quantity to a different unit.

```scrut
$ wo 'UnitConvert[Quantity[1, "Kilometers"], "Meters"]'
Quantity[1000, Meters]
```

With one argument, converts to SI base units.

```scrut
$ wo 'UnitConvert[Quantity[1, "Liters"]]'
Quantity[1/1000, Meters^3]
```
