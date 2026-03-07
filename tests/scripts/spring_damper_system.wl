
(* Spring-damper (mass-spring-damper) system: m*x'' + c*x' + k*x = 0 *)
(* Parameters *)
m = 1;    (* mass *)
k = 10;   (* spring constant *)

(* Show underdamped, critically damped, and overdamped cases *)
cUnder = 2;    (* underdamped: c < 2*Sqrt[k*m] ≈ 6.32 *)
cCrit = 2 Sqrt[k m] // N;  (* critically damped *)
cOver = 12;    (* overdamped *)

(* Solve the ODE for each case *)
solUnder = DSolve[{m x''[t] + cUnder x'[t] + k x[t] == 0, x[0] == 1, x'[0] == 0}, x[t], t][[1]];
solCrit = DSolve[{m x''[t] + cCrit x'[t] + k x[t] == 0, x[0] == 1, x'[0] == 0}, x[t], t][[1]];
solOver = DSolve[{m x''[t] + cOver x'[t] + k x[t] == 0, x[0] == 1, x'[0] == 0}, x[t], t][[1]];

Plot[
  {x[t] /. solUnder, x[t] /. solCrit, x[t] /. solOver},
  {t, 0, 8},
  PlotStyle -> {
    {Thick, RGBColor[0.2, 0.5, 0.9]},
    {Thick, RGBColor[0.9, 0.4, 0.1]},
    {Thick, RGBColor[0.1, 0.7, 0.3]}
  },
  PlotLegends -> Placed[{
    Style["Underdamped (ζ = 0.32)", 12],
    Style["Critically damped (ζ = 1.0)", 12],
    Style["Overdamped (ζ = 1.90)", 12]
  }, {0.65, 0.75}],
  AxesLabel -> {Style["Time (s)", 14], Style["Displacement x(t)", 14]},
  PlotLabel -> Style["Spring-Damper System Response\nm x'' + c x' + k x = 0,  x(0)=1,  x'(0)=0", 15, Bold],
  PlotRange -> {-0.6, 1.1},
  GridLines -> Automatic,
  GridLinesStyle -> Directive[GrayLevel[0.85]],
  ImageSize -> 650,
  Filling -> {1 -> {Axis, Directive[Opacity[0.05], RGBColor[0.2, 0.5, 0.9]]}},
  Epilog -> {
    Dashed, GrayLevel[0.5], Line[{{0, 0}, {8, 0}}]
  }
]
