# `OpenWrite`

Opens a file for writing.

```scrut
$ wo 'Head[OpenWrite[]]'
OutputStream
```

A file name starting with `!` opens a pipe into an external command instead:
what is written to the stream becomes the command's standard input, and
closing the stream is what tells the command its input has ended.

```scrut
$ wo 'stream = OpenWrite["!tr a-z A-Z"]; WriteString[stream, "hello pipe\n"]; Close[stream];'
HELLO PIPE
Null
```
