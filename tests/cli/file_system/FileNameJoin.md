# `FileNameJoin`

Joins path components using the platform path separator.

```scrut
$ wo 'FileNameJoin[{"a", "b", "c"}]'
a/b/c
```

A single string is a path already; joining it normalises the separators.

```scrut
$ wo 'FileNameJoin["/a//b/c.wlx/"]'
/a/b/c.wlx
```

A component may carry its own separators; a run of them collapses to one.

```scrut
$ wo 'FileNameJoin[{"/a", "b/", "c.wlx"}]'
/a/b/c.wlx
```

```scrut
$ wo 'FileNameJoin[{"//a", "/b"}]'
/a/b
```
