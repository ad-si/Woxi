# `StringReplace`

Applies one or more replacement rules to a string.

```scrut
$ wo 'StringReplace["Hello world", "world" -> "moon"]'
Hello moon
```

`$1`… expand into the replacement before it is evaluated, so they reach string
literals inside a compound right-hand side:

```scrut
$ wo 'StringReplace["abc", RegularExpression["(b)"] :> "<" <> "$1" <> ">"]'
a<b>c
```
