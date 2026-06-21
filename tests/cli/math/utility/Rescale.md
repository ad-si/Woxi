# `Rescale`

Rescales values to a specified range.

```scrut
$ wo 'Rescale[3, {1, 5}]'
1/2
```

`Rescale[list]` uses the global minimum and maximum across all elements,
preserving nested structure.

```scrut
$ wo 'Rescale[{{1, 2}, {3, 4}}]'
{{0, 1/3}, {2/3, 1}}
```
