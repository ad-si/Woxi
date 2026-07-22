# `ListPlay`

Plays a list of amplitude levels as a sound. The amplitudes are normalized so
the minimum maps to `-1` and the maximum to `+1`, then sampled at `SampleRate`
(8000 Hz by default). The result is a `Sound` object, so it reports
`Head -> Sound` and — in the visual hosts (the Woxi Playground and Woxi Studio)
— renders a playable audio widget.

```scrut
$ wo 'Head[ListPlay[{0.1, 0.2, 0.3, -0.1}]]'
Sound
```

```scrut
$ wo 'Head[ListPlay[Table[Sin[2 Pi 50 t], {t, 0, 1, 1./2000}]]]'
Sound
```
