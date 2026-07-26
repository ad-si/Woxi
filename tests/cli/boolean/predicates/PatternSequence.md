# `PatternSequence`

Represents several arguments in a row, spliced into the pattern that contains
it:

```scrut
$ wo 'MatchQ[f[1, 2, 3], f[PatternSequence[1, 2], 3]]'
True
```

Naming it binds the whole sequence:

```scrut
$ wo 'f[1, 2] /. f[x : PatternSequence[_, _]] -> {x}'
{1, 2}
```

Repeating it repeats the group, so only an even number of arguments matches a
pair:

```scrut
$ wo '{MatchQ[f[1, 2, 3, 4], f[PatternSequence[_, _] ..]], MatchQ[f[1, 2, 3], f[PatternSequence[_, _] ..]]}'
{True, False}
```
