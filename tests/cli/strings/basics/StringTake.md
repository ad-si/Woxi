# `StringTake`

```scrut
$ wo 'StringTake["Hello World!", 5]'
Hello
```

A span `i;;j` takes characters `i` through `j`.

```scrut
$ wo 'StringTake["hello", 2;;4]'
ell
```

```scrut
$ wo 'StringTake["hello", ;;-2]'
hell
```

A three-element specification `{m, n, s}` is a span with step `s`:

```scrut
$ wo 'StringTake["abcde", {1, 5, 2}]'
ace
```

A list of integers that cannot be read as a span is re-read as a list of
*separate* specifications, one result per entry. A step of `0` makes the span
reading impossible, so `{1, 5, 0}` means "take 1, take 5, take 0":

```scrut
$ wo 'StringTake["abcde", {1, 5, 0}]'
{a, abcde, }
```

Four or more entries can never be a span, so they are always read that way:

```scrut
$ wo 'StringTake["abcde", {1, 2, 3, 4}]'

StringTake::ambgsntx: Warning: interpreting list of integers as a list of sequence specifications.
{a, ab, abc, abcd}
```

An empty specification takes nothing:

```scrut
$ wo 'StringTake["abcde", {}]'

```
