# `RandomSample`

Random selection without replacement.

```scrut
$ wo 'Length[RandomSample[{a, b, c}]]'
3
```

Asking for more elements than the set holds is reported, with a pointer to the
function that does allow repeats:

```scrut
$ wo 'RandomSample[{1, 2}, 5]'

RandomSample::smplen: RandomSample cannot generate a sample of length 5, which is greater than the length of the sample set {1, 2}. If you want a choice of possibly repeated elements from the set, use RandomChoice.
RandomSample[{1, 2}, 5]
```
