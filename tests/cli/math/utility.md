# Utility Math Functions

Indicator functions, sampling (`RandomInteger`), `Reap`/`Sow`,
and the membership predicate `MemberQ`.

## `Unitize`

```scrut
$ wo 'Unitize[{0, 1, -3, 0, 5}]'
{0, 1, 1, 0, 1}
```

## `Ramp`

```scrut
$ wo 'Ramp[{-2, -1, 0, 1, 2}]'
{0, 0, 0, 1, 2}
```

## `KroneckerDelta`

```scrut
$ wo 'KroneckerDelta[1, 1]'
1
```

```scrut
$ wo 'KroneckerDelta[1, 2]'
0
```

## `UnitStep`

```scrut
$ wo 'UnitStep[{-1, 0, 1}]'
{0, 1, 1}
```

## `Reap` / `Sow`

```scrut
$ wo 'Reap[Sow[1]; Sow[2]; 42]'
{42, {{1, 2}}}
```

```scrut
$ wo 'Reap[42]'
{42, {}}
```



## `MemberQ`

Checks if an element is in a list.

```scrut
$ wo 'MemberQ[{1, 2}, 2]'
True
```

```scrut
$ wo 'MemberQ[{1, 2}, 3]'
False
```


## `RandomInteger`

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


