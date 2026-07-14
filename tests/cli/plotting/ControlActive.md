# `ControlActive`

`ControlActive[active, normal]` represents an object that is shown as
`active` while it sits inside a control that is being *actively* manipulated
(e.g. while a slider thumb is being dragged), and as `normal` at all other
times. It is used to substitute a cheaper computation during dragging.

Outside of an interactive notebook nothing is ever actively manipulated, so
`ControlActive` always evaluates to its `normal` (inactive) form — matching
`wolframscript` in script mode.

```scrut
$ wo 'ControlActive[1, 2]'
2
```

The chosen argument is evaluated as usual:

```scrut
$ wo 'ControlActive[1 + 1, 2 + 2]'
4
```

```scrut
$ wo 'ControlActive["fast", "slow"]'
slow
```

A common pattern is picking a coarse plot while dragging and a fine one once
the control settles:

```scrut
$ wo 'ControlActive[10, 100]'
100
```

With no arguments `ControlActive[]` queries whether a control is currently
being actively manipulated. Outside an interactive notebook nothing is, so it
evaluates to `False`:

```scrut
$ wo 'ControlActive[]'
False
```
