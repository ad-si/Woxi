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

Sign predicates test the magnitude.

```scrut
$ wo 'Positive[Quantity[5, "Meters"]]'
True
```

`Mod` of two compatible quantities returns the result in the divisor's unit.

```scrut
$ wo 'Mod[Quantity[7, "Meters"], Quantity[300, "Centimeters"]]'
Quantity[100, Centimeters]
```

A compound unit with a negative exponent is written as a quotient, the way
the unit itself reads:

```scrut
$ wo 'ToString[InputForm[Quantity[1, "Meters"/"Seconds"]]]'
Quantity[1, "Meters"/"Seconds"]
```

```scrut
$ wo 'ToString[InputForm[Quantity[1, "Meters"/("Seconds"*"Kilograms")]]]'
Quantity[1, "Meters"/("Kilograms"*"Seconds")]
```

A unit that is only a negative power keeps that form, with the exponent
parenthesised:

```scrut
$ wo 'ToString[InputForm[Quantity[1, "Seconds"^-1]]]'
Quantity[1, "Seconds"^(-1)]
```

`ToString` spells the unit out, using the singular for a magnitude of 1:

```scrut
$ wo 'ToString[Quantity[3, "Meters"]]'
3 meters
```

```scrut
$ wo 'ToString[Quantity[1, "Feet"]]'
1 foot
```

`TextString` abbreviates it instead:

```scrut
$ wo 'TextString[Quantity[2.5, "Kilograms"]]'
2.5 kg
```

A currency or percent attaches directly to the magnitude:

```scrut
$ wo 'TextString[Quantity[3, "USDollars"]]'
$3
```

```scrut
$ wo 'TextString[Quantity[1, "Percent"]]'
1%
```

A compound unit reads as a phrase — and an exponent forces the spelled-out
form even for `TextString`:

```scrut
$ wo 'ToString[Quantity[3, "Meters"/"Seconds"]]'
3 meters per second
```

```scrut
$ wo 'TextString[Quantity[3, "Kilometers"/"Hours"]]'
3 km/h
```

```scrut
$ wo 'TextString[Quantity[3, "Meters"/"Seconds"^2]]'
3 meters per second squared
```

A list is rendered element-wise:

```scrut
$ wo 'ToString[{Quantity[1, "Meters"], Quantity[2, "Feet"]}]'
{1 meter, 2 feet}
```
