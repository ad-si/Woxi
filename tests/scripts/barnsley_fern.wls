BarnsleyFern[{x_, y_}] := Module[{},
   i = RandomInteger[{1, 100}];
   If[i <= 1, {xt = 0, yt = 0.16*y},
    If[i <= 8, {xt = 0.2*x - 0.26*y, yt = 0.23*x + 0.22*y + 1.6},
     If[i <= 15, {xt = -0.15*x + 0.28*y, yt = 0.26*x + 0.24*y + 0.44},
      {xt = 0.85*x + 0.04*y, yt = -0.04*x + 0.85*y + 1.6}]]];
   {xt, yt}];
points = NestList[BarnsleyFern, {0,0}, 100000];
Show[Graphics[{Hue[.35, 1, .7], PointSize[.001], Point[#] & /@ points}]]
