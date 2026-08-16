# `FileNames`

Returns a list of file names matching a pattern in the current directory.

```scrut
$ wo 'ListQ[FileNames["*"]]'
True
```

The current directory is the one `SetDirectory` last set,
not the one the script was started from.

```scrut
$ wo 'dir = CreateDirectory[]; Export[FileNameJoin[{dir, "hello.txt"}], "x"]; SetDirectory[dir]; FileNames[]'
{hello.txt}
```
