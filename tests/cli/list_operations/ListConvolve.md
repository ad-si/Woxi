# `ListConvolve`

Computes discrete convolution of two lists.

```scrut
$ wo 'ListConvolve[{1, 2}, {3, 4}]'
{10}
```

A padding array tiles outwards from the edges of a two-dimensional list:

```scrut
$ wo 'ListConvolve[{{1, 2}, {3, 4}}, {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, 1, {{10, 20}, {30, 40}}]'
{{291, 244, 257}, {167, 23, 33}, {219, 53, 63}}
```
