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

The same message prints at most three times per calculation, after which a
`General::stop` notice replaces it. Later ones are not merely silent — they
are not recorded either, so `$MessageList` never grows past those four
entries however long the loop runs.

```scrut
$ wo 'Do[1/0, {50}]; Length[$MessageList]'
* (glob*)
4
```

Quiet does not exempt a message from that count; it only stops it printing.

```scrut
$ wo 'Quiet[Do[1/0, {50}]; Length[$MessageList]]'
4
```
