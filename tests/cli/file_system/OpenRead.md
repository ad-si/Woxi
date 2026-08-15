# `OpenRead`

Opens a file for reading.

A file name starting with `!` names an external command instead of a file:
the command is run through the shell and the stream reads its standard output.

```scrut
$ wo 'stream = OpenRead["!echo woxi"]; line = ReadLine[stream]; Close[stream]; line'
woxi
```

Reading past the command's output yields `EndOfFile`:

```scrut
$ wo 'stream = OpenRead["!echo woxi"]; ReadLine[stream]; rest = ReadLine[stream]; Close[stream]; rest'
EndOfFile
```
