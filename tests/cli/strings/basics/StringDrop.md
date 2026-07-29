# `StringDrop`

```scrut
$ wo 'StringDrop["Hello World!", 6]'
World!
```

An empty specification drops nothing:

```scrut
$ wo 'StringDrop["abcde", {}]'
abcde
```

Unlike `StringTake`, `StringDrop` accepts only span specifications — a list of
integers that is not one is refused rather than re-read:

```scrut
$ wo 'StringDrop["abcde", {1, 2, 3, 4}]'

StringDrop::seqs: Sequence specification (+n, -n, {+n}, {-n}, {m, n} or {m, n, s}) expected at position 2 in StringDrop[abcde, {1, 2, 3, 4}].
StringDrop[abcde, {1, 2, 3, 4}]
```
