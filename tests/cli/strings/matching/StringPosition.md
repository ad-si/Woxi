# `StringPosition`

Finds all positions of a substring, returning {start, end} pairs (1-indexed).

```scrut
$ wo 'StringPosition["abcabc", "bc"]'
{{2, 3}, {5, 6}}
```

```scrut
$ wo 'StringPosition["hello", "l"]'
{{3, 3}, {4, 4}}
```
