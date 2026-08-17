# `DeleteStopwords`

Removes common stopwords ("the", "a", "of", ...) from a string, keeping the
surrounding punctuation of the remaining words.

```scrut
$ wo 'DeleteStopwords["A long time ago, in a galaxy far, far away"]'
long time ago, galaxy far, far away
```

Given a list of words, stopword elements are dropped from the list.

```scrut
$ wo 'DeleteStopwords[{"a", "list", "of", "words"}]'
{list, words}
```

Given an association, stopword keys are dropped.

```scrut
$ wo 'DeleteStopwords[WordCounts["the cat sat on the mat"]]'
<|mat -> 1, sat -> 1, cat -> 1|>
```
