semiPrimeQ[n_Integer] := Module[{factors, numfactors},
  factors = FactorInteger[n] // Transpose;
  numfactors = factors[[2]] // Total  ;
  numfactors == 2
  ]


(* ========== *)

semiPrimeQ[#] & /@ Range[100];
Position[%, True] // Flatten
