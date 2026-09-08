# `FileNameSplit`

Splits a path into its components.

```scrut
$ wo 'FileNameSplit["a/b/c.txt"]'
{a, b, c.txt}
```

Only *trailing* separators are dropped. The leading empty piece is what marks
a path absolute, and an interior one is kept too.

```scrut
$ wo 'FileNameSplit["/a//b/"]'
{, a, , b}
```
