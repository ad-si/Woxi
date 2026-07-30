# `StringPart`

Extracts a character by 1-based index.

```scrut
$ wo 'StringPart["Hello", 2]'
e
```

A specification that is not a position at all is a different complaint from
one that simply is not there:

```scrut
$ wo 'StringPart["abcde", 1.5]'

StringPart::pkspec1: The expression 1.5 cannot be used as a part specification.
StringPart[abcde, 1.5]
```

```scrut
$ wo 'StringPart["abcde", 10]'

StringPart::partw: Part 10 of abcde does not exist.
StringPart[abcde, 10]
```

Inside a list, anything unusable is reported as the whole list not existing:

```scrut
$ wo 'StringPart["abcde", {1, x}]'

StringPart::partw: Part {1, x} of abcde does not exist.
StringPart[abcde, {1, x}]
```
