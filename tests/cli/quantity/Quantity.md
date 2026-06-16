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
