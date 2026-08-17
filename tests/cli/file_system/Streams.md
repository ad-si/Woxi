# `Streams`

List currently open streams. wolframscript also opens an internal temporary
script file before evaluating `-code`, so its stream list contains a third
`OutputStream[/...WolframScriptTemporary/...]` entry that Woxi does not
have. Match the always-present stdout/stderr prefix and tolerate any extra
trailing streams.

```scrut
$ wo 'Streams[]'
\{OutputStream\[stdout, 1\], OutputStream\[stderr, 2\].*\} (regex)
```

The two always-open ones are also available as `$StandardOutputStream` and
`$StandardErrorStream`:

```scrut
$ wo '$StandardOutputStream'
OutputStream[stdout, 1]
```

```scrut
$ wo '$StandardErrorStream'
OutputStream[stderr, 2]
```

```scrut
$ wo '{$StandardOutputStream, $StandardErrorStream} === Take[Streams[], 2]'
True
```
