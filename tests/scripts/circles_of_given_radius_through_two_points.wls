Off[Solve::ratnz];
circs::invrad = "The radius is invalid.";
circs::equpts = "The given points (`1`, `2`) are equivalent.";
circs::dist =
  "The given points (`1`, `2`) and (`3`, `4`) are too far apart for \
radius `5`.";
circs[_, _, 0.] := Message[circs::invrad];
circs[{p1x_, p1y_}, {p1x_, p1y_}, _] :=
  Message[circs::equpts, p1x, p1y];
circs[p1 : {p1x_, p1y_}, p2 : {p2x_, p2y_}, r_] /;
  EuclideanDistance[p1, p2] > 2 r :=
 Message[circs::dist, p1x, p1y, p2x, p2y, r];
circs[p1 : {p1x_, p1y_}, p2 : {p2x_, p2y_}, r_] :=
 Values /@
  Solve[Abs[x - p1x]^2 + Abs[y - p1y]^2 ==
    Abs[x - p2x]^2 + Abs[y - p2y]^2 == r^2, {x, y}];
