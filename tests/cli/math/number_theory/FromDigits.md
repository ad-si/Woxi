# `FromDigits`

Constructs an integer from its digits.

```scrut
$ wo 'FromDigits[{1, 2, 3, 4, 5}]'
12345
```

With base 2 (binary):

```scrut
$ wo 'FromDigits[{1, 1, 1, 1, 1, 1, 1, 1}, 2]'
255
```
