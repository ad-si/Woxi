# `MatchQ`

Tests whether an expression matches a pattern.

```scrut
$ wo 'MatchQ[5, _Integer]'
True
```

```scrut
$ wo 'MatchQ["hi", _Integer]'
False
```
