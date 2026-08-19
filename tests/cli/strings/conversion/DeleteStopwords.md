# `DeleteStopwords`

Removes common stopwords ("the", "a", "of", ...) from a string. Only the
words go: the spaces and punctuation around them stay put, so every deleted
word leaves a gap behind.

```scrut
$ wo 'DeleteStopwords["A long time ago, in a galaxy far, far away"]'
 long time ago,   galaxy far, far away
```

Given a list of words, stopword elements are dropped from the list.

```scrut
$ wo 'DeleteStopwords[{"a", "list", "of", "words"}]'
{list, words}
```

Anything else — an association of word counts, a number — is an error.

```scrut
$ wo 'DeleteStopwords[WordCounts["the cat sat on the mat"]]'

DeleteStopwords::strse: A string or list of strings is expected at position 1 in DeleteStopwords[<|the -> 2, mat -> 1, on -> 1, sat -> 1, cat -> 1|>].
DeleteStopwords[<|the -> 2, mat -> 1, on -> 1, sat -> 1, cat -> 1|>]
```
