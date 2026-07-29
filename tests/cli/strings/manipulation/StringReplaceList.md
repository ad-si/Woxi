# `StringReplaceList`

Returns every single-replacement variant of a string.

```scrut
$ wo 'StringReplaceList["abcabc", "a" -> "X"]'
{Xbcabc, abcXbc}
```

Matches may overlap, since each result replaces exactly one of them:

```scrut
$ wo 'StringReplaceList["aaa", "aa" -> "X"]'
{Xa, aX}
```

A pattern rule contributes one result per span it can match — every length at
every start position, longest first for a greedy pattern:

```scrut
$ wo 'StringReplaceList["abcd", __ -> "X"]'
{X, Xd, Xcd, Xbcd, aX, aXd, aXcd, abX, abXd, abcX}
```

A list of rules is tried in its written order within each start position:

```scrut
$ wo 'StringReplaceList["abcabc", {"a" -> "1", "c" -> "2"}]'
{1bcabc, ab2abc, abc1bc, abcab2}
```

The third argument caps the number of results:

```scrut
$ wo 'StringReplaceList["abcd", __ -> "X", 3]'
{X, Xd, Xcd}
```

A replacement that is not a string keeps the result as a `StringExpression`:

```scrut
$ wo 'StringReplaceList["abc", "b" -> 5]'
{StringExpression[a, 5, c]}
```

A second argument that is not a replacement rule is refused, naming the last
offending element:

```scrut
$ wo 'StringReplaceList["abc", {"a", "b"}]'

StringReplaceList::srep: b is not a valid string replacement rule.
StringReplaceList[abc, {a, b}]
```
