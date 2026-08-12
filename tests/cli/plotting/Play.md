# `Play`

Builds a sound object whose amplitude is the given function of the time
variable (in seconds). The result is a `Sound` object, so it reports
`Head -> Sound`, prints as `-Sound-`, and — in the visual hosts (the Woxi
Playground and Woxi Studio) — renders a playable audio widget.

```scrut
$ wo 'Play[Sin[2 Pi 440 t], {t, 0, 1}]'
-Sound-
```

```scrut
$ wo 'Head[Play[Sin[2 Pi 440 t], {t, 0, 1}]]'
Sound
```

Options follow the time iterator. `SampleRate` sets how many samples per
second the amplitude function is synthesized at (8000 by default):

```scrut
$ wo 'Head[Play[Sin[2 Pi 440 t], {t, 0, 1}, SampleRate -> 2^13]]'
Sound
```

An argument after the iterator that is not an option leaves the expression
unevaluated:

```scrut
$ wo 'Play[Sin[t], {t, 0, 1}, 5]'
Play[Sin[t], {t, 0, 1}, 5]
```
