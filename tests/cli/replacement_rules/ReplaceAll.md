# `ReplaceAll` (`/.`)

Replaces parts of an expression according to rules.

```scrut
$ wo '{a, b} /. a -> x'
{x, b}
```

```scrut
$ wo '{x^2, y^2} /. x -> 2'
{4, y^2}
```
