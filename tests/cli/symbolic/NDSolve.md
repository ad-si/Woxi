# `NDSolve`

Solves ordinary differential equations numerically.

```scrut
$ wo "NDSolve[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 5}]"
{{y -> InterpolatingFunction[{{0., 5.}}, <>]}}
```

The equation does not have to be linear in the function being solved for —
here the pendulum equation, whose `Sin[y[t]]` has no closed-form solution:

```scrut
$ wo "s = NDSolve[{y''[t] == -Sin[y[t]], y[0] == 1, y'[0] == 0}, y, {t, 0, 5}]; (y /. s[[1]])[3.0]"
-0.94875159694288
```

A solution rule carries an `InterpolatingFunction`, which can be
differentiated:

```scrut
$ wo "s = Flatten[NDSolve[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 5}]]; s[[1]][[2]]'[2.0]"
-0.1353352846442526
```

Substituting the solution into a derivative works the same way, which is how
a function and its slope are sampled together (for a phase portrait, say):

```scrut
$ wo "s = NDSolve[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 5}]; {y[2.0], y'[2.0]} /. First[s]"
{0.1353352832380283, -0.1353352846442526}
```
