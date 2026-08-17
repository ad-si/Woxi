# `WriteString`

Write a string to an output stream.

The stream can be named directly, or given as the `OutputStream[…]`
expression that `$StandardOutputStream` stands for. Both write to the
process's standard output, and neither appends a newline of its own.

```scrut
$ wo 'WriteString["stdout", "hello world\n"]'
hello world
Null
```

```scrut
$ wo 'WriteString[$StandardOutputStream, "hello world\n"]'
hello world
Null
```

`$StandardErrorStream` writes to standard error instead:

```scrut
$ wo 'WriteString[$StandardErrorStream, "oh no\n"]' 2> /dev/null
Null
```

All arguments after the stream are written in one go, without separators:

```scrut
$ wo 'WriteString[$StandardOutputStream, "a", "b", "c\n"]'
abc
Null
```
