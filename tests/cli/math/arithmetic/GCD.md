# `GCD`

### `GCD[12, 8]`

The GCD of 12 and 8 is 4.

```scrut
$ wo 'GCD[12, 8]'
4
```


### `GCD[48, 18]`

The GCD of 48 and 18 is 6.

```scrut
$ wo 'GCD[48, 18]'
6
```


### `GCD[100, 50]`

The GCD of 100 and 50 is 50.

```scrut
$ wo 'GCD[100, 50]'
50
```


### `GCD[17, 19]`

The GCD of 17 and 19 is 1 (coprime numbers).

```scrut
$ wo 'GCD[17, 19]'
1
```


### `GCD[0, 5]`

The GCD of 0 and any number n is |n|.

```scrut
$ wo 'GCD[0, 5]'
5
```


### `GCD[15, 25, 35]`

The GCD of 15, 25, and 35 is 5.

```scrut
$ wo 'GCD[15, 25, 35]'
5
```


### `GCD[24, 36, 60]`

The GCD of 24, 36, and 60 is 12.

```scrut
$ wo 'GCD[24, 36, 60]'
12
```


### `GCD[-12, 8]`

The GCD works with negative numbers (GCD of -12 and 8 is 4).

```scrut
$ wo 'GCD[-12, 8]'
4
```


### `GCD[21, 14]`

The GCD of 21 and 14 is 7.

```scrut
$ wo 'GCD[21, 14]'
7
```


### `GCD[7 + 3 I, 2]`

GCD also works over the Gaussian integers, returning the canonical associate.

```scrut
$ wo 'GCD[7 + 3 I, 2]'
1 + I
```
