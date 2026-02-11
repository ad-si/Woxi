# Functions

## `Directory[]`

```scrut
$ wo 'Directory[]'
/\S+ (regex)
```

```scrut
$ wo 'StringQ[Directory[]]'
True
```

## `CreateFile[]`

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
