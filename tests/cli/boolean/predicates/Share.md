# `Share`

Memory sharing optimization. wolframscript returns the number of bytes
shared; Woxi has no shared-memory optimisation and always returns `0`. The
return value is therefore an integer in both backends — match any digit
sequence.

```scrut
$ wo 'Share[]'
\d+ (regex)
```
