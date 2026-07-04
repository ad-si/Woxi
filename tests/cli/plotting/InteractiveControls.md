# Interactive Manipulation Controls

The control and animation heads from the
[InteractiveManipulation](https://reference.wolfram.com/language/guide/InteractiveManipulation.html)
guide render as interactive widgets inside a Jupyter notebook or the
[playground](/). In script mode (like `wolframscript -code`) they have no
notebook to live in, so they stay unevaluated as their canonical form rather
than producing an interactive object.

Animation heads:

```scrut
$ wo 'Animate[x^2, {x, 0, 5}]'
Animate[x^2, {x, 0, 5}]
```

```scrut
$ wo 'ListAnimate[{1, 2, 3}]'
ListAnimate[{1, 2, 3}]
```

Selection-bar controls:

```scrut
$ wo 'SetterBar[1, {1, 2, 3}]'
SetterBar[1, {1, 2, 3}]
```

```scrut
$ wo 'CheckboxBar[{1}, {1, 2, 3}]'
CheckboxBar[{1}, {1, 2, 3}]
```

```scrut
$ wo 'TogglerBar[{1}, {1, 2, 3}]'
TogglerBar[{1}, {1, 2, 3}]
```

```scrut
$ wo 'RadioButton[1]'
RadioButton[1]
```

Progress, trigger, and two-dimensional / interval sliders:

```scrut
$ wo 'ProgressIndicator[0.4]'
ProgressIndicator[0.4]
```

```scrut
$ wo 'IntervalSlider[{2, 4}]'
IntervalSlider[{2, 4}]
```

```scrut
$ wo 'Slider2D[{0, 0}]'
Slider2D[{0, 0}]
```
