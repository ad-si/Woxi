IsPrime[n_Integer] := Block[{},
  If[n <= 1, Return[False]];
  If[n == 2, Return[True]]; If[Mod[n, 2] == 0, Return[False]];
  For[k = 3, k <= Sqrt[n], k += 2, If[Mod[n, k] == 0, Return[False]]];
  Return[True]]
