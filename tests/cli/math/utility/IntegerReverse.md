# `IntegerReverse`

Reverses the digits of an integer.

```scrut
$ wo 'IntegerReverse[0]'
0
```

The base has to be greater than 1:

```scrut
$ wo 'IntegerReverse[123, 1]'

IntegerReverse::ibmr: Positive integer greater than 1 or mixed radix specification expected at position 2 of IntegerReverse[123, 1].
IntegerReverse[123, 1]
```
