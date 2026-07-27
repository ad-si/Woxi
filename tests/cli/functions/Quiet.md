# `Quiet`

Evaluates an expression while suppressing its messages.

```scrut
$ wo 'Quiet[1/0]'
ComplexInfinity
```

A quieted message is left out of `$MessageList`, which otherwise lists every
message raised so far in the calculation.

```scrut
$ wo 'Quiet[1/0]; $MessageList'
{}
```
