# `AudioPlot`

Plots the waveform of an audio object as amplitude over time (in seconds).
Accepts `Audio` objects built from sample data as well as `Sound` objects
synthesized with `Play`.

```scrut
$ wo 'Head[AudioPlot[Audio[{0, 0.5, 1, 0.5, 0}]]]'
Graphics
```

```scrut
$ wo 'Head[AudioPlot[Play[Sin[440 2 Pi t], {t, 0, 0.1}]]]'
Graphics
```
