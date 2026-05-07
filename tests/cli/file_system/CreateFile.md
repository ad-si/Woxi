# `CreateFile`

Creates an empty file on disk, returning the resolved path.
With no argument, an auto-generated name in the temporary
directory is used.

```scrut
$ wo 'CreateFile[]'
/\S+ (regex)
```

```scrut
$ wo 'CreateFile["_delete_me_"]'
.+_delete_me_ (regex)
```

```scrut
$ wo 'a = CreateFile[]; a'
/\S+ (regex)
```
