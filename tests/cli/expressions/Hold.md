# `Hold`

Prevents its argument from being evaluated.

```scrut
$ wo 'Hold[1 + 2]'
Hold[1 + 2]
```

Taking a part out of the wrapper lifts it out of the hold, so it evaluates.

```scrut
$ wo 'Hold[1 + 2][[1]]'
3
```

```scrut
$ wo 'First[Hold[1 + 2, 3 + 4]]'
3
```

Parts that keep the `Hold` head stay held.

```scrut
$ wo 'Rest[Hold[1 + 2, 3 + 4]]'
Hold[3 + 4]
```
