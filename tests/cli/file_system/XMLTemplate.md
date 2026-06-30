# `XMLTemplate`

`XMLTemplate[src]` yields a `TemplateObject` for an XML/HTML template. Like
`StringTemplate`, but slots are inserted with `HTMLFragment`. The template may
contain `` `slot` `` markers as well as embedded `<* expr *>` expressions, where
`#key` / `#n` refer to the template's slots.

A template combining a slot and an embedded expression:

```scrut
$ wo 'XMLTemplate["Range of `a`: <* Range[#a] *>."]'
TemplateObject[{Range of , TemplateSlot[a], : , TemplateExpression[Range[TemplateSlot[a]]], .}, CombinerFunction -> StringJoin, InsertionFunction -> HTMLFragment, MetaInformation -> <||>]
```

Applying arguments fills the slot and evaluates the expression:

```scrut
$ wo 'TemplateApply[XMLTemplate["Range of `a`: <* Range[#a] *>."], Association["a" -> 5]]'
Range of 5: {1, 2, 3, 4, 5}.
```
