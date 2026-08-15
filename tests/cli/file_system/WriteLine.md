# `WriteLine`

Write a string to an output stream, followed by a newline.
It is `WriteString` plus a line terminator.

The channel is an open stream, a file name, or one of the standard streams
`"stdout"` and `"stderr"`:

```scrut
$ wo 'WriteLine["stdout", "first"]; WriteLine["stdout", "second"]'
first
second
Null
```

Writing through a stream opened with `OpenWrite` appends one line per call,
so the file holds the lines in the order they were written:

```scrut
$ wo 'file = FileNameJoin[{$TemporaryDirectory, "woxi-write-line.txt"}]; stream = OpenWrite[file]; WriteLine[stream, "Some log text"]; WriteLine[stream, "more"]; Close[stream]; ReadList[file, String]'
{Some log text, more}
```

A file name starting with `!` writes into an external command's standard
input, the same as for `WriteString`:

```scrut
$ wo 'stream = OpenWrite["!tr a-z A-Z"]; WriteLine[stream, "hello pipe"]; Close[stream];'
HELLO PIPE
Null
```
