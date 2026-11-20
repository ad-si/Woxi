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

A `WhenEvent` may watch a real-valued expression instead of an equation: the
event fires where the expression crosses zero. Here the integration stops
once `y` has decayed to `1/4`, at `t = Log[4]`:

```scrut
$ wo "s = NDSolve[{y'[t] == -y[t], y[0] == 1, WhenEvent[y[t] - 1/4, \"StopIntegration\"]}, y, {t, 0, 5}]; Round[10^4 (y[\"Domain\"] /. First[s])]"
{{0, 13863}}
```

### Step-wise solving

`NDSolve`ProcessEquations` sets a problem up as an `NDSolve`StateData`
object without solving it. `NDSolve`Reinitialize` restarts it from other
initial conditions, `NDSolve`Iterate` integrates it up to a given time (here
backward, from `t == 1` to `t == -2`) and `NDSolve`ProcessSolutions` gives
the solution rules:

```scrut
$ wo "s = First[NDSolve\`ProcessEquations[{x'[t] == y[t], y'[t] == -x[t], x[0] == 1, y[0] == 0}, {x, y}, t]]; s2 = First[NDSolve\`Reinitialize[s, {x[1] == 0, y[1] == 1}]]; NDSolve\`Iterate[s2, -2]; Chop[({x[-2], y[-2]} /. NDSolve\`ProcessSolutions[s2]) - {Sin[-3.], Cos[-3.]}, 10^-5]"
{0, 0}
```
