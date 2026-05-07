# `Hash`

Computes a hash of an expression.
The one-argument form uses an implementation-defined hash whose numeric
value differs between Woxi and Mathematica, so prefer an explicit algorithm:

```scrut
$ wo 'Hash["hello", "MD5", "HexString"]'
5d41402abc4b2a76b9719d911017c592
```
