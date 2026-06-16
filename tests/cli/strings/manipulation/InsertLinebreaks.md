# `InsertLinebreaks`

Inserts line breaks into a string so that each line is at most a given number
of characters (default 78). Words are kept whole where they fit and packed
greedily; a word longer than the width is hard-broken.

```scrut
$ wo 'InsertLinebreaks["abcdefgh", 3]'
abc
def
gh
```

Wrapping happens at spaces when possible.

```scrut
$ wo 'InsertLinebreaks["hello world foo bar", 7]'
hello
world
foo bar
```

A string already shorter than the width is returned unchanged.

```scrut
$ wo 'InsertLinebreaks["abc", 5]'
abc
```
