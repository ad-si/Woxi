squareFree[n_Integer] := DeleteCases[Last /@ FactorInteger[n], 1] === {};
findSquareFree[n__] := Select[Range[n], squareFree];
findSquareFree[45]
findSquareFree[10^9, 10^9 + 145]
Length[findSquareFree[10^6]]
