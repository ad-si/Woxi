# `Check`

Returns its first argument unless a message is emitted, in which case
it returns the second argument.

```scrut
$ wo 'Check[1/2, error]'
1/2
```

A message counts as soon as it is generated. `General::stop` keeps the
fourth and later ones from being printed, but `Check` still reacts to them —
here the outer `Quiet` keeps the three printed ones out of the way:

```scrut
$ wo 'Quiet[Table[Check[1/0, "F"], {6}]]'
{F, F, F, F, F, F}
```

A `Quiet` inside the `Check` does stop it; one outside does not:

```scrut
$ wo 'Check[Quiet[1/0], "F"]'
ComplexInfinity
```

```scrut
$ wo 'Quiet[Check[1/0, "F"]]'
F
```

`Check` reacts to messages, not to control flow, so a `Throw` passes
straight through it:

```scrut
$ wo 'Catch[Check[Throw[1], "F"]]'
1
```
