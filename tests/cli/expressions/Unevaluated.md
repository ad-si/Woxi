# `Unevaluated`

A marker that tells the evaluator to pass its argument to the surrounding
function without first evaluating it.
Wrappers like `List` leave `Unevaluated` alone.

```scrut
$ wo 'List[Unevaluated[1 + 2], 3]'
{Unevaluated[1 + 2], 3}
```
