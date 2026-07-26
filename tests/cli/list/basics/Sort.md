# `Sort`

Sorts a list in ascending order.

```scrut
$ wo 'Sort[{3, 1, 4, 1, 5, 9, 2, 6}]'
{1, 1, 2, 3, 4, 5, 6, 9}
```

```scrut
$ wo 'Sort[{5, 2, 8, 1, 9}]'
{1, 2, 5, 8, 9}
```

```scrut
$ wo 'Sort[{1}]'
{1}
```

```scrut
$ wo 'Sort[{}]'
{}
```

```scrut
$ wo 'Sort[{-5, 3, 0, -2, 7}]'
{-5, -2, 0, 3, 7}
```

```scrut
$ wo 'Sort[{3.14, 2.71, 1.41, 2.23}]'
{1.41, 2.23, 2.71, 3.14}
```

```scrut
$ wo 'Sort[{10, 5, 15, 5, 20}]'
{5, 5, 10, 15, 20}
```

```scrut
$ wo 'Sort[{-10, -5, -15, -20}]'
{-20, -15, -10, -5}
```

With an ordering function a pair is left as it is unless the comparison is a
definite `False`, which swaps it — so the elements a comparison cannot separate
come out reversed:

```scrut
$ wo 'Sort[Range[5], Mod[#1, 2] > Mod[#2, 2] &]'
{5, 3, 1, 4, 2}
```

A symbolic comparison leaves the order alone:

```scrut
$ wo 'Sort[{c, a, b}, Less]'
{c, a, b}
```
