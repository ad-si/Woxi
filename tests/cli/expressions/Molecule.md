# `Molecule`

Represents a chemical molecule as a symbolic expression of atoms and bonds.
The canonical form is `Molecule[{atoms…}, {bonds…}, {metadata…}]`. An atom is a
bare element symbol, or `Atom["El", "Prop" -> value, …]` when it carries extra
properties. A molecule can be constructed from a chemical name (whose hydrogens
are materialized as explicit atoms):

```scrut
$ wo 'Molecule["water"]'
Molecule[{O, H, H}, {Bond[{1, 2}, Single], Bond[{1, 3}, Single]}, {}]
```

… from a [SMILES](https://en.wikipedia.org/wiki/SMILES) string
(organic-subset hydrogens stay implicit):

```scrut
$ wo 'Molecule["C=O"]'
Molecule[{C, O}, {Bond[{1, 2}, Double]}, {}]
```

… or from explicit atom and bond lists (also kept implicit):

```scrut
$ wo 'Molecule[{"C", "O"}, {Bond[{1, 2}, "Single"]}]'
Molecule[{C, O}, {Bond[{1, 2}, Single]}, {}]
```

Aromatic rings, formal charges, and isotopes are supported. A bracket-atom
hydrogen count rides along as a `HydrogenCount` property:

```scrut
$ wo 'Molecule["[NH4+]"]'
Molecule[{Atom[N, FormalCharge -> 1, HydrogenCount -> 4]}, {}, {}]
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

## `BondQ`

`BondQ[mol, bond]` tests whether `bond` is a bond of `mol`. The atom pair is
unordered, and it indexes the same hydrogen-complete atom list `AtomList`
returns — so ethanol's implicit hydrogens (atoms 4 to 9) are addressable:

```scrut
$ wo 'BondQ[Molecule["CCO"], Bond[{2, 1}]]'
True
```

```scrut
$ wo 'BondQ[Molecule["CCO"], Bond[{1, 4}]]'
True
```

```scrut
$ wo 'BondQ[Molecule["CCO"], Bond[{1, 3}]]'
False
```

Omitting the bond type matches a bond of any type; giving one requires an
exact match:

```scrut
$ wo 'BondQ[Molecule["c1ccccc1"], Bond[{1, 2}, "Aromatic"]]'
True
```

```scrut
$ wo 'BondQ[Molecule["c1ccccc1"], Bond[{1, 2}, "Single"]]'
False
```

## `ConnectedMoleculeQ` and `ConnectedMoleculeComponents`

`ConnectedMoleculeQ[mol]` tests whether the atoms of `mol` are all joined by
bonds. A `.` in a SMILES string separates fragments, so a salt is not
connected:

```scrut
$ wo 'ConnectedMoleculeQ[Molecule["ethanol"]]'
True
```

```scrut
$ wo 'ConnectedMoleculeQ[Molecule["[Na+].[Cl-]"]]'
False
```

`ConnectedMoleculeComponents[mol]` returns each connected fragment as its own
molecule, with the bonds renumbered against the fragment's atoms:

```scrut
$ wo 'ConnectedMoleculeComponents[Molecule["CC.O"]]'
{Molecule[{C, C}, {Bond[{1, 2}, Single]}, {}], Molecule[{O}, {}, {}]}
```

Both take any specification `Molecule` accepts, not just a molecule:

```scrut
$ wo 'ConnectedMoleculeQ["CCO"]'
True
```

## `MoleculeValue`

Computes properties of a molecule
(`"AtomCount"`, `"BondCount"`, `"AtomList"`, `"BondList"`,
`"MolecularFormula"`, and `"NetCharge"`). The molecular formula is returned as a
typeset `Row` of element symbols with the counts as subscripts:

```scrut
$ wo 'MoleculeValue[Molecule["caffeine"], "MolecularFormula"]'
Subscript[C, 8]Subscript[H, 10]Subscript[N, 4]Subscript[O, 2]
```

Properties can also be accessed by applying the molecule to a property name:

```scrut
$ wo 'Molecule["benzene"]["AtomCount"]'
12
```

## Structure diagram

In visual hosts (the playground and Woxi Studio) a molecule displays as its 2-D
structure diagram. The same diagram is available anywhere via
`ExportString[mol, "SVG"]`, a standalone SVG document (with an XML declaration):

```scrut
$ wo 'StringTake[ExportString[Molecule["caffeine"], "SVG"], 4]'
<?xm
```

It is a structure drawing, not a text summary, so it carries no `Formula: `
label:

```scrut
$ wo 'StringContainsQ[ExportString[Molecule["water"], "SVG"], "Formula: "]'
False
```

## `MoleculePlot`

`MoleculePlot[mol]` draws the full 2-D skeletal structure diagram as a
`Graphics` object. Atoms are laid out with a spring embedder; single, double,
triple, and aromatic bonds, heteroatom labels, formal charges, and isotope
mass numbers are all drawn. It accepts a molecule or any specification
`Molecule` itself accepts:

```scrut
$ wo 'MoleculePlot["caffeine"]'
-Graphics-
```

Bonds are drawn as vector polylines rather than `<line>` elements, matching the
markup wolframscript emits, so a benzene diagram contains no `<line>`:

```scrut
$ wo 'StringCount[ExportString[MoleculePlot["benzene"], "SVG"], "<line"]'
0
```
