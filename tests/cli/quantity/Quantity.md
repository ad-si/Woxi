# `Quantity`

Represents a physical quantity with a unit.

```scrut
$ wo 'Quantity[2, 3]'
6
```

Singular unit names are canonicalized to their plural form.

```scrut
$ wo 'Quantity[1, "Mole"]'
Quantity[1, Moles]
```

When quantities are multiplied, the compound unit's factors are ordered
alphabetically.

```scrut
$ wo 'Quantity[100, "Watts"] * Quantity[2, "Hours"]'
Quantity[200, Hours*Watts]
```

Functions like `Abs` and `Floor` apply to the magnitude and keep the unit.

```scrut
$ wo 'Abs[Quantity[-5, "Meters"]]'
Quantity[5, Meters]
```

`Max` and `Min` compare quantities after converting to a common unit, returning
the winner in its original unit.

```scrut
$ wo 'Max[Quantity[1, "Meters"], Quantity[50, "Centimeters"]]'
Quantity[1, Meters]
```
