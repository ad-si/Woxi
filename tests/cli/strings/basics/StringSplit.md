# `StringSplit`

```scrut
$ wo 'StringSplit["Hello World!", " "]'
{Hello, World!}
```

With a list of delimiters, leading and trailing empty pieces are dropped:

```scrut
$ wo 'StringSplit["a1b2c3", {"1", "2", "3"}]'
{a, b, c}
```

An explicit maximum keeps empty pieces and the original remainder:

```scrut
$ wo 'StringSplit["a1b2c3", {"1", "2", "3"}, 2]'
{a, b2c3}
```
