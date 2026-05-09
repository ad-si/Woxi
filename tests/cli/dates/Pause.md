# `Pause`

`Pause[n]` pauses evaluation for at least `n` seconds (wall-clock time)
and then returns `Null`.

```scrut
$ wo 'Pause[0]'
Null
```

`AbsoluteTiming` confirms that `Pause` actually waits:

```scrut
$ wo 'AbsoluteTiming[Pause[0.2]][[1]] >= 0.2'
True
```

The result of `Pause` is always `Null`:

```scrut
$ wo 'AbsoluteTiming[Pause[0.1]][[2]]'
Null
```
