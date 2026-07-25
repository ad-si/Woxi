# `TemplateApply`

Fills the slots of a template string.

```scrut
$ wo 'TemplateApply["x = `x`", <|"x" -> 1|>]'
x = 1
```

A `StringTemplate` object may be given instead of the raw text:

```scrut
$ wo 'TemplateApply[StringTemplate["Hello, my name is ``."], {"Bob"}]'
Hello, my name is Bob.
```

`<*expr*>` is an expression slot, evaluated inline.
With no arguments to fill, the one-argument form still runs them:

```scrut
$ wo 'TemplateApply["x <*2^3*> y"]'
x 8 y
```
