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

Compound base units are grouped to render unambiguously.

```scrut
$ wo 'UnitConvert[Quantity[1, "Bars"]]'
Quantity[100000, Kilograms/(Meters*Seconds^2)]
```

```scrut
$ wo 'UnitConvert[Quantity[1, "Newtons"]]'
Quantity[1, (Kilograms*Meters)/Seconds^2]
```

Area units convert to the SI base `Meters^2`.

```scrut
$ wo 'UnitConvert[Quantity[1, "Acres"]]'
Quantity[316160658/78125, Meters^2]
```

```scrut
$ wo 'UnitConvert[Quantity[1, "Hectares"], "SquareMeters"]'
Quantity[10000, Meters^2]
```
