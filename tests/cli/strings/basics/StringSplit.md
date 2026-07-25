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

A rule keeps the delimiters between the pieces, and a named pattern binds
the matched text:

```scrut
$ wo 'StringSplit["a1b2c3", x_?LetterQ :> ToUpperCase[x]]'
{A, 1, B, 2, C, 3}
```

Each rule in a list gets its own replacement; a bare delimiter inserts
nothing:

```scrut
$ wo 'StringSplit["a-b_c", {"-", "_" -> "="}]'
{a, b, =, c}
```
