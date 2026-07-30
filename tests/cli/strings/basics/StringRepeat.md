# `StringRepeat`

Repeats a string n times.

```scrut
$ wo 'StringRepeat["ab", 3]'
ababab
```

```scrut
$ wo 'StringRepeat["x", 5]'
xxxxx
```

```scrut
$ wo 'StringRepeat["hello", 0]'

```

Zero repetitions give the empty string, and a negative count is reported:

```scrut
$ wo 'StringRepeat["abcde", 0] === ""'
True
```

```scrut
$ wo 'StringRepeat["abcde", -1]'

StringRepeat::intp: Positive integer expected at position 2 in StringRepeat[abcde, -1].
StringRepeat[abcde, -1]
```

The third argument truncates, and is checked the same way:

```scrut
$ wo 'StringRepeat["abcde", 3, 7]'
abcdeab
```

```scrut
$ wo 'StringRepeat["abcde", 2, -1]'

StringRepeat::intp: Positive integer expected at position 3 in StringRepeat[abcde, 2, -1].
StringRepeat[abcde, 2, -1]
```
