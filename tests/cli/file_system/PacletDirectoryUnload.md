# `PacletDirectoryUnload`

Unregisters paclet directories and returns the ones still loaded.

```scrut
$ wo 'PacletDirectoryLoad[Directory[]]; PacletDirectoryUnload[Directory[]]'
{}
```

Unloading a directory that was never loaded changes nothing:

```scrut
$ wo 'PacletDirectoryLoad[Directory[]]; PacletDirectoryUnload["elsewhere"] === {Directory[]}'
True
```
