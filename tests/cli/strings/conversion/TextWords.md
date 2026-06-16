# `TextWords`

Splits a string into words, ignoring punctuation.

```scrut
$ wo 'TextWords["Hello, world! How are you?"]'
{Hello, world, How, are, you}
```

Internal hyphens are kept as part of the word.

```scrut
$ wo 'TextWords["the YT-1300 droid"]'
{the, YT-1300, droid}
```

A second argument returns only the first n words.

```scrut
$ wo 'TextWords["first second third fourth", 2]'
{first, second}
```
