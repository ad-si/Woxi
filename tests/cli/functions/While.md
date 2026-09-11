# `While`

`While[test, body]` loop.

```scrut
$ wo 'i = 0; While[i < 5, i++]; i'
5
```

The body runs only while the test is `True`. A test that gives anything
else — a number, an unassigned symbol — ends the loop without a message.

```scrut
$ wo 'i = 0; While[i, i++]; i'
0
```

```scrut
$ wo 'While[q, 1]'
Null
```
