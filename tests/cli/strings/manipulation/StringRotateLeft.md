# `StringRotateLeft`

Rotates a string to the left by `n` characters.

```scrut
$ wo 'StringRotateLeft["abcdef", 2]'
cdefab
```

The count has to be an integer:

```scrut
$ wo 'StringRotateLeft["abcde", 1.5]'

StringRotateLeft::int: Integer expected at position 2 in StringRotateLeft[abcde, 1.5].
StringRotateLeft[abcde, 1.5]
```
