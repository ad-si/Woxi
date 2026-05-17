crule[m_, b_] := Module[{d = Det[m], a},
  Table[a = m; a[[All, k]] = b; Det[a]/d, {k, Length[m]}]]

crule[{
  {2, -1, 5, 1},
  {3, 2, 2, -6},
  {1, 3, 3, -1},
  {5, -2, -3, 3}
 } , {-3, -32, -47, 49}]
