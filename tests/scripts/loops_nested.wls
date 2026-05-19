m = {{1, 2, 3, 4}, {5, 6, 7, 20}, {9, 10, 11, 12}};
Do[
  Print[m[[i, j]]];
  If[m[[i, j]] === 20, Return[]],
  {i, 1, Dimensions[m][[1]]},
  {j, 1, Dimensions[m][[2]]}]
