# `Range`

Generates a sequence of numbers.

### Range[n]

Generates {1, 2, ..., n}.

```scrut
$ wo 'Range[5]'
{1, 2, 3, 4, 5}
```

```scrut
$ wo 'Range[1]'
{1}
```

```scrut
$ wo 'Range[0]'
{}
```

```scrut
$ wo 'Range[10]'
{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
```

### Range[min, max]

Generates {min, min+1, ..., max}.

```scrut
$ wo 'Range[3, 7]'
{3, 4, 5, 6, 7}
```

```scrut
$ wo 'Range[0, 5]'
{0, 1, 2, 3, 4, 5}
```

```scrut
$ wo 'Range[-3, 2]'
{-3, -2, -1, 0, 1, 2}
```

```scrut
$ wo 'Range[5, 5]'
{5}
```

### Range[min, max, step]

Generates {min, min+step, ..., max}.

```scrut
$ wo 'Range[1, 10, 2]'
{1, 3, 5, 7, 9}
```

```scrut
$ wo 'Range[0, 20, 5]'
{0, 5, 10, 15, 20}
```

```scrut
$ wo 'Range[10, 1, -1]'
{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}
```

```scrut
$ wo 'Range[5, -5, -2]'
{5, 3, 1, -1, -3, -5}
```

```scrut
$ wo 'Range[1, 10, 3]'
{1, 4, 7, 10}
```
