points = 5000;
a = {0, 0};
b = {1, 0};
c = {0.5, 1};
d = {.7, .3};
S = {};
For[i = 1, i < points, i++, t = RandomInteger[2];
 If[t == 0, d = Mean[{a, d}],
  If[t == 1, d = Mean[{b, d}], d = Mean[{c, d}]]]; AppendTo[S, d]]
Graphics[Point[S]]
