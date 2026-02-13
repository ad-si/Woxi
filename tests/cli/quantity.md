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

## Compound Unit Conversion

Convert between compound units:

```scrut
$ wo 'UnitConvert[Quantity[1, "Meters"/"Seconds"], "Kilometers"/"Hours"]'
Quantity[18/5, Kilometers/Hours]
```

```scrut
$ wo 'UnitConvert[Quantity[7500, "Kilometers"/"Seconds"], "Kilometers"/"Hours"]'
Quantity[27000000, Kilometers/Hours]
```

## Unit Abbreviations

Common unit abbreviations are supported:

```scrut
$ wo 'Quantity[1, "km/h"]'
Quantity[1, Kilometers/Hours]
```

```scrut
$ wo 'Quantity[1, "m/s"]'
Quantity[1, Meters/Seconds]
```

```scrut
$ wo 'Quantity[1, "mph"]'
Quantity[1, Miles/Hours]
```

```scrut
$ wo 'UnitConvert[Quantity[1, "Meters"/"Seconds"], "km/h"]'
Quantity[18/5, Kilometers/Hours]
```

## SpeedOfLight

```scrut
$ wo 'UnitConvert[Quantity[1, "SpeedOfLight"], "Meters"/"Seconds"]'
Quantity[299792458, Meters/Seconds]
```

```scrut
$ wo 'UnitConvert[Quantity[1, "SpeedOfLight"], "km/h"]'
Quantity[5396264244/5, Kilometers/Hours]
```

## Compound Unit Simplification

Same-dimension units are simplified during arithmetic:

```scrut
$ wo 'Quantity[100, "Kilometers"/"Hours"] / Quantity[3.2, "Seconds"]'
Quantity[0.008680555555555556, Kilometers/Seconds^2]
```
