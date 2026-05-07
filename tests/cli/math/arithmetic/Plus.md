# `Plus`

Add numbers.

```scrut
$ wo 'Plus[1, 2]'
3
```

```scrut
$ wo 'Plus[1, 2, 3]'
6
```

```scrut
$ wo 'Plus[-1, -2]'
-3
```


### `+`

```scrut
$ wo '1+2'
3
```

```scrut
$ wo '8 + (-5)'
3
```


### List arithmetic (threading)

Arithmetic operations are automatically threaded over lists:

```scrut
$ wo '{1, 2, 3} + 2'
{3, 4, 5}
```

```scrut
$ wo '2 + {1, 2, 3}'
{3, 4, 5}
```

```scrut
$ wo '{1, 2, 3} + {4, 5, 6}'
{5, 7, 9}
```

```scrut
$ wo '{1, 2, 3} - 1'
{0, 1, 2}
```

```scrut
$ wo '{1, 2, 3} * 2'
{2, 4, 6}
```

```scrut
$ wo '{2, 4, 6} / 2'
{1, 2, 3}
```

```scrut
$ wo '{1, 2, 3} ^ 2'
{1, 4, 9}
```
