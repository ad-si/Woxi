findTaxi[n_] := Sort[Keys[Select[Counts[Flatten[Table[x^3 + y^3, {x, 1, n}, {y, x, n}]]], GreaterThan[1]]]];
Print[Take[findTaxi[100], 25]]
