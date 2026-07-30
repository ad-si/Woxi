# `Options`

Returns or sets the options associated with a symbol.

The core symbolic and numeric functions report their built-in defaults:

```scrut
$ wo 'Options[Total]'
{AllowedHeads -> Automatic, Method -> Automatic}
```

```scrut
$ wo 'Options[Factor]'
{Extension -> None, GaussianIntegers -> False, Modulus -> 0, Trig -> False}
```

An option whose default is a global stays a delayed rule:

```scrut
$ wo 'Options[Series]'
{Analytic -> True, Assumptions :> $Assumptions, SeriesTermGoal -> Automatic}
```

A second argument selects one option:

```scrut
$ wo 'Options[Simplify, TimeConstraint]'
{TimeConstraint -> 300}
```

A symbol with no options reports none:

```scrut
$ wo 'Options[Sin]'
{}
```

`SetOptions` replaces entries in the list and returns the updated one:

```scrut
$ wo 'SetOptions[Total, Method -> "X"]'
{AllowedHeads -> Automatic, Method -> X}
```

The change sticks, and `OptionValue` reads it back:

```scrut
$ wo 'SetOptions[Total, Method -> 7]; OptionValue[Total, Method]'
7
```

A name that is not already an option refuses the whole call, so nothing is
changed even when other names in it are valid:

```scrut
$ wo 'SetOptions[Total, Bogus -> 1, Method -> 9]'

SetOptions::optnf: Bogus is not a known option for Total.
SetOptions[Total, Bogus -> 1, Method -> 9]
```
