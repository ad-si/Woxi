# `HighpassFilter`

Apply a highpass filter to data. The exact last digit of each filtered
sample is a property of the underlying f64 math library (libm in Woxi vs.
the kernel's internal implementation in wolframscript) so the expected
output is given as a regex tolerant of that drift.

```scrut
$ wo 'HighpassFilter[{1, 2, 3, 4, 5}, 0.3]'
\{0\.71987930582788\d\d, 1\.565448565047395\d?, 2\.359894452882456\d?, 3\.15434034071751\d\d?, 3\.99990959993702\d\d?\} (regex)
```
