# `UniqueElements`

Given a list of lists, returns for each list the elements that are unique to it
(i.e. that appear in that list but in none of the others). Duplicates within a
list are removed and the first-appearance order is preserved.

```scrut
$ wo 'UniqueElements[{{1, 2, 3, 4, 5}, {3, 4, 5, 6, 7}}]'
{{1, 2}, {6, 7}}
```

```scrut
$ wo 'UniqueElements[{{1, 2, 2, b, b, a}, {4, 3, 2, 1}}]'
{{b, a}, {4, 3}}
```

```scrut
$ wo 'UniqueElements[{{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}}]'
{{1}, {}, {6}}
```

An optional test function decides when two elements are considered equivalent:

```scrut
$ wo 'UniqueElements[{{1, 2}, {4, 6}}, Mod[#1, 3] == Mod[#2, 3] &]'
{{2}, {6}}
```
