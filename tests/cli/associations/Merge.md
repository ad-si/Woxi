# `Merge`

```scrut
$ wo 'Merge[{<|a -> 1|>, <|a -> 2, b -> 3|>}, Total]'
<|a -> 3, b -> 3|>
```

Anything in the list that is not an association or a rule is named:

```scrut
$ wo 'Merge[{<|"a" -> 1|>, 7}, Total]'

Merge::list1: The argument 7 is not a valid list of Associations or rules or lists of rules.
Merge[{<|a -> 1|>, 7}, Total]
```
