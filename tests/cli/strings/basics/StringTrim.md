# `StringTrim`

Removes leading and trailing whitespace.

```scrut
$ wo 'StringTrim["  hello  "]'
hello
```

```scrut
$ wo 'StringTrim["  hello world  "]'
hello world
```

With a pattern, removes leading and trailing occurrences:

```scrut
$ wo 'StringTrim["xxhelloxx", "xx"]'
hello
```
