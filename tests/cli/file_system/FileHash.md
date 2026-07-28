# `FileHash`

The hash of a file's bytes, with the same algorithms and output formats
`Hash` takes. Without them it is the MD5 as an integer.

```scrut
$ wo 'Export["a.txt", "abc", "Text"]; FileHash["a.txt"]'
191415658344158766168031473277922803570
```

```scrut
$ wo 'Export["b.txt", "abc", "Text"]; FileHash["b.txt", "SHA256", "HexString"]'
ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
```

A file that cannot be read reports it and fails:

```scrut
$ wo 'FileHash["no-such-file-here.txt"]'

FileHash::noopen: Cannot open .*no-such-file-here.txt. (regex)
$Failed
```
