# `SubscriptBox`

Represents a subscript box for typesetting.

A string may carry inline `\!\(\*SubscriptBox[…]\)` linear syntax. In
OutputForm the box segment shows as `DisplayForm[…]`, and the prose around
it is kept:

```scrut
$ wo 'Print["\!\(\*SubscriptBox[\(p\), \(0\)]\) is the tested value"]'
DisplayForm[SubscriptBox[p, 0]] is the tested value
Null
```

The string itself is untouched — the markers are content:

```scrut
$ wo 'StringLength["\!\(\*SubscriptBox[\(p\), \(0\)]\)"]'
26
```
