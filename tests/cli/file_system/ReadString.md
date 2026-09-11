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

The terminator may be a string pattern instead of a literal string. The read
returns the text before the first match:

```scrut
$ wo 'ReadString[StringToStream["aaa%--%HEAD%--%rest"], "%" ~~ Repeated["-", {1, 10}] ~~ "%HEAD%" ~~ Repeated["-", {1, 10}] ~~ "%"]'
aaa
```

An anchor or a repeat on its own is not a terminator, though the same
construct inside an alternative is:

```scrut
$ wo 'ReadString[StringToStream["a\nb"], EndOfLine | "\r"]'
a
```

Nothing matching gives everything that is left, and a message:

```scrut
$ wo 'ReadString[StringToStream["abc"], StartOfLine ~~ "zzz"]'

ReadString::notfound: Specified terminator not found.
abc
```

Option rules may follow the terminator:

```scrut
$ wo 'ReadString[StringToStream["a-b"], "-", TimeConstraint -> 10]'
a
```
