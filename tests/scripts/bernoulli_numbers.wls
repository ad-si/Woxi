bernoulli[n_] := Module[{a = ConstantArray[0, n + 2]},
  Do[
    a[[m]] = 1/m;
    If[m == 1 && a[[1]] != 0, Print[{m - 1, a[[1]]}]];
    Do[
     a[[j - 1]] = (j - 1)*(a[[j - 1]] - a[[j]]);
     If[j == 2 && a[[1]] != 0, Print[{m - 1, a[[1]]}]];
     , {j, m, 2, -1}];
    , {m, 1, n + 1}];
  ]
bernoulli[20]
