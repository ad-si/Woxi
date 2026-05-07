# `RandomInteger`

### `RandomInteger[]`

Randomly gives 0 or 1.

```scrut
$ wo 'MemberQ[{0, 1}, RandomInteger[]]'
True
```


### `RandomInteger[{1, 6}]`

Randomly gives a number between 1 and 6.

```scrut
$ wo 'MemberQ[{1, 2, 3, 4, 5, 6}, RandomInteger[{1, 6}]]'
True
```


### `RandomInteger[{1, 6}, 50]`

Randomly gives 50 numbers between 1 and 6.

```scrut
$ wo 'AllTrue[RandomInteger[{1, 6}, 50], 1 <= # <= 6 &]'
True
```
