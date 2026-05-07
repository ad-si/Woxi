# `Rest`

Returns the list without its first element.

```scrut
$ wo 'Rest[{1, 2, 3}]'
{2, 3}
```

```scrut
$ wo 'Rest[{5, 10, 15, 20}]'
{10, 15, 20}
```

```scrut
$ wo 'Rest[{a, b, c}]'
{b, c}
```

```scrut
$ wo 'Rest[{42}]'
{}
```

```scrut
$ wo 'Rest[{-5, 0, 5}]'
{0, 5}
```

```scrut
$ wo 'Rest[{1.5, 2.5, 3.5}]'
{2.5, 3.5}
```
