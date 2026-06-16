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
