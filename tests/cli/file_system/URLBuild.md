# `URLBuild`

Joins URL path segments with `/`.

```scrut
$ wo 'URLBuild[{"a", "b"}]'
a/b
```

An association of `URLParse` components is assembled back into a URL:

```scrut
$ wo 'URLBuild[<|"Scheme" -> "https", "Domain" -> "x.com", "Path" -> {"", "p"}, "Query" -> {"k" -> "v"}|>]'
https://x.com/p?k=v
```

```scrut
$ wo 'URLBuild[URLParse["https://x.com/p?q=1"]]'
https://x.com/p?q=1
```
