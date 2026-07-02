# `Molecule`

Represents a chemical molecule as a symbolic expression of atoms and bonds.
A molecule can be constructed from a chemical name:

```scrut
$ wo 'Molecule["water"]'
Molecule[{Atom[O], Atom[H], Atom[H]}, {Bond[{1, 2}, Single], Bond[{1, 3}, Single]}]
```

… from a [SMILES](https://en.wikipedia.org/wiki/SMILES) string
(hydrogen atoms are added to fill the normal valences):

```scrut
$ wo 'Molecule["C=O"]'
Molecule[{Atom[C], Atom[O], Atom[H], Atom[H]}, {Bond[{1, 2}, Double], Bond[{1, 3}, Single], Bond[{1, 4}, Single]}]
```

… or from explicit atom and bond lists:

```scrut
$ wo 'Molecule[{"C", "O"}, {Bond[{1, 2}, "Single"]}]'
Molecule[{Atom[C], Atom[O], Atom[H], Atom[H], Atom[H], Atom[H]}, {Bond[{1, 2}, Single], Bond[{1, 3}, Single], Bond[{1, 4}, Single], Bond[{1, 5}, Single], Bond[{2, 6}, Single]}]
```

Aromatic rings, formal charges, and isotopes are supported:

```scrut
$ wo 'Molecule["[NH4+]"]'
Molecule[{Atom[N, FormalCharge -> 1], Atom[H], Atom[H], Atom[H], Atom[H]}, {Bond[{1, 2}, Single], Bond[{1, 3}, Single], Bond[{1, 4}, Single], Bond[{1, 5}, Single]}]
```

## `MoleculeQ`

Tests whether an expression is a valid molecule:

```scrut
$ wo 'MoleculeQ[Molecule["ethanol"]]'
True
```

```scrut
$ wo 'MoleculeQ[5]'
False
```

## `AtomList` and `BondList`

Return the atoms and bonds of a molecule:

```scrut
$ wo 'AtomList[Molecule["methane"]]'
{Atom[C], Atom[H], Atom[H], Atom[H], Atom[H]}
```

```scrut
$ wo 'BondList[Molecule["ammonia"]]'
{Bond[{1, 2}, Single], Bond[{1, 3}, Single], Bond[{1, 4}, Single]}
```

## `MoleculeValue`

Computes properties of a molecule
(`"AtomCount"`, `"BondCount"`, `"AtomList"`, `"BondList"`,
`"MolecularFormula"`, and `"NetCharge"`):

```scrut
$ wo 'MoleculeValue[Molecule["caffeine"], "MolecularFormula"]'
C8H10N4O2
```

Properties can also be accessed by applying the molecule to a property name:

```scrut
$ wo 'Molecule["benzene"]["AtomCount"]'
12
```
