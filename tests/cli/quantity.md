# Quantity

## Construction

Construct a quantity with magnitude and unit:

```scrut
$ wo 'Quantity[3, "Meters"]'
Quantity[3, Meters]
```

One-argument form assumes magnitude 1:

```scrut
$ wo 'Quantity["Meters"]'
Quantity[1, Meters]
```

Compound units:

```scrut
$ wo 'Quantity[60, "Miles"/"Hours"]'
Quantity[60, Miles/Hours]
```

```scrut
$ wo 'Quantity[1, "Meters"/"Seconds"^2]'
Quantity[1, Meters/Seconds^2]
```

## Arithmetic

Same-unit addition:

```scrut
$ wo 'Quantity[3, "Meters"] + Quantity[2, "Meters"]'
Quantity[5, Meters]
```

Cross-unit addition (converts to first unit):

```scrut
$ wo 'Quantity[3, "Meters"] + Quantity[2, "Kilometers"]'
Quantity[2003, Meters]
```

Subtraction:

```scrut
$ wo 'Quantity[3, "Meters"] - Quantity[1, "Meters"]'
Quantity[2, Meters]
```

Scalar multiplication:

```scrut
$ wo '2 * Quantity[3, "Meters"]'
Quantity[6, Meters]
```

Quantity × Quantity:

```scrut
$ wo 'Quantity[3, "Meters"] * Quantity[2, "Seconds"]'
Quantity[6, Meters*Seconds]
```

Division:

```scrut
$ wo 'Quantity[10, "Meters"] / Quantity[2, "Seconds"]'
Quantity[5, Meters/Seconds]
```

Power:

```scrut
$ wo 'Quantity[3, "Meters"]^2'
Quantity[9, Meters^2]
```

## UnitConvert

```scrut
$ wo 'UnitConvert[Quantity[1, "Kilometers"], "Meters"]'
Quantity[1000, Meters]
```

```scrut
$ wo 'UnitConvert[Quantity[1, "Hours"], "Seconds"]'
Quantity[3600, Seconds]
```

```scrut
$ wo 'UnitConvert[Quantity[1, "Miles"], "Kilometers"]'
Quantity[25146/15625, Kilometers]
```

## Accessors

```scrut
$ wo 'QuantityMagnitude[Quantity[5, "Meters"]]'
5
```

```scrut
$ wo 'QuantityMagnitude[Quantity[1, "Kilometers"], "Meters"]'
1000
```

```scrut
$ wo 'QuantityUnit[Quantity[5, "Meters"]]'
Meters
```

## Predicates

```scrut
$ wo 'QuantityQ[Quantity[5, "Meters"]]'
True
```

```scrut
$ wo 'QuantityQ[5]'
False
```

```scrut
$ wo 'CompatibleUnitQ[Quantity[1, "Meters"], Quantity[1, "Kilometers"]]'
True
```

```scrut
$ wo 'CompatibleUnitQ[Quantity[1, "Meters"], Quantity[1, "Seconds"]]'
False
```

## Comparisons

```scrut
$ wo 'Quantity[5, "Meters"] > Quantity[3, "Meters"]'
True
```

```scrut
$ wo 'Quantity[1, "Kilometers"] > Quantity[500, "Meters"]'
True
```

```scrut
$ wo 'Quantity[5, "Meters"] == Quantity[5, "Meters"]'
True
```
