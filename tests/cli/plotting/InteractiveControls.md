# Interactive Manipulation Controls

The control and animation heads from the
[InteractiveManipulation](https://reference.wolfram.com/language/guide/InteractiveManipulation.html)
guide render as interactive widgets inside a Jupyter notebook or the
[playground](/). In script mode (like `wolframscript -code`) they have no
notebook to live in, so they stay unevaluated as their canonical form rather
than producing an interactive object.

Animation heads stay unevaluated as their canonical form. (`wolframscript`
instead expands them into a large internal `Manipulate[…]` object — and
`ListAnimate` even into one containing a non-deterministic gensym symbol — so
these are documentation only and not part of the conformance sweep; they are
covered by the `control_wrappers_stay_symbolic_without_warning` unit test.)

```wolfram
Animate[x^2, {x, 0, 5}]
(* Woxi: Animate[x^2, {x, 0, 5}] *)

ListAnimate[{1, 2, 3}]
(* Woxi: ListAnimate[{1, 2, 3}] *)
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

Standalone animator and the draggable / clickable panes (these auto-play or
respond to pointer input in the Playground and Studio; in script mode they stay
symbolic):

```scrut
$ wo 'Animator[{0, 10}]'
Animator[{0, 10}]
```

```scrut
$ wo 'LocatorPane[Dynamic[p], Graphics[Point[p]]]'
LocatorPane[Dynamic[p], -Graphics-]
```

```scrut
$ wo 'ClickPane[Graphics[{}], f]'
ClickPane[-Graphics-, f]
```
