# Deeply nested result rendering

Rendering a deeply nested result must not overflow the stack. The interpreter
runs on a large-stack worker thread so nesting depths that wolframscript renders
(here a 3000-deep list) are handled rather than crashing.

```scrut
$ wo 'StringLength[ToString[Nest[List, {1}, 3000]]]'
6003
```

A deep symbolic tower renders too:

```scrut
$ wo 'StringLength[ToString[Nest[f, x, 3000]]]'
9001
```
