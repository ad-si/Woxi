# `Reap`

Collects expressions sown by `Sow` during evaluation.

```scrut
$ wo 'Reap[Sow[1]; Sow[2]; 42]'
{42, {{1, 2}}}
```

```scrut
$ wo 'Reap[42]'
{42, {}}
```
