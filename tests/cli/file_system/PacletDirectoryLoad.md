# `PacletDirectoryLoad`

Registers directories in which the paclet manager looks for paclets and
returns all loaded paclet directories.

A directory may be a paclet itself — a directory holding a `PacletInfo.wl`
file — or a directory collecting several of them. Once it is loaded,
`Needs` finds the contexts its paclets declare in their `"Kernel"`
extensions.

Nothing is loaded in a fresh session:

```scrut
$ wo 'PacletDirectoryLoad[]'
{}
```

Relative directories are reported in absolute form:

```scrut
$ wo 'PacletDirectoryLoad[Directory[]] === {Directory[]}'
True
```

Loading the same directory twice does not list it twice:

```scrut
$ wo 'PacletDirectoryLoad[Directory[]]; PacletDirectoryLoad[Directory[]] === {Directory[]}'
True
```

A directory that does not exist is reported and not loaded:

```scrut
$ wo 'PacletDirectoryLoad["no-such-paclet-directory"]'

PacletDirectoryLoad::nodir: Directory .*no-such-paclet-directory not found. (regex)
{}
```
