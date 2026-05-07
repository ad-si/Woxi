# `Block`

Like `Module`, but temporarily rebinds *global* symbols rather than
introducing fresh locals. Changes are undone when `Block` returns.

```scrut
$ wo 'Block[{x = 10}, x^2]'
100
```
