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
