# `ReadString`

Read the contents of a file or stream as a string.

```scrut
$ wo 'ReadString[x]'
ReadString[x]
```

Reading a stream takes everything left in it:

```scrut
$ wo 'ReadString[StringToStream["hi there"]]'
hi there
```

A second argument gives a terminator, so repeated reads yield the fields
between separators:

```scrut
$ wo 's = StringToStream["a-b-c"]; {ReadString[s, "-"], ReadString[s, "-"]}'
{a, b}
```

A stream with nothing left gives `EndOfFile`:

```scrut
$ wo 'ReadString[StringToStream[""]]'
EndOfFile
```
