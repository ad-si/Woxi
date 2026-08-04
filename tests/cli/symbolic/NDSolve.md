# `NDSolve`

Solves ordinary differential equations numerically. The answer is a rule
giving an `InterpolatingFunction` over the requested range:

```scrut
$ wo "s = NDSolve[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 5}]; {Head[y /. First[s]], (y /. First[s])[\"Domain\"]}"
{InterpolatingFunction, {{0., 5.}}}
```

Sampling it gives the solution. The step sizes an adaptive solver picks are
its own, so the digits below are the ones Woxi and wolframscript share
rather than every digit either prints — here `E^-2`:

```scrut
$ wo "s = NDSolve[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 5}]; (y /. First[s])[2.0]"
0\.135335\d+ (regex)
```

The equation does not have to be linear in the function being solved for —
here the pendulum equation, whose `Sin[y[t]]` has no closed-form solution:

```scrut
$ wo "s = NDSolve[{y''[t] == -Sin[y[t]], y[0] == 1, y'[0] == 0}, y, {t, 0, 5}]; (y /. s[[1]])[3.0]"
-0\.9487515\d+ (regex)
```

A solution rule carries an `InterpolatingFunction`, which can be
differentiated:

```scrut
$ wo "s = Flatten[NDSolve[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 5}]]; s[[1]][[2]]'[2.0]"
-0\.13533\d+ (regex)
```

Substituting the solution into a derivative works the same way, which is how
a function and its slope are sampled together (for a phase portrait, say):

```scrut
$ wo "s = NDSolve[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 5}]; {y[2.0], y'[2.0]} /. First[s]"
\{0\.135335\d+, -0\.13533\d+\} (regex)
```
