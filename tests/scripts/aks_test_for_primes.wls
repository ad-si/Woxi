Print["powers of (x-1)"]
(x - 1)^( Range[0, 7]) // Expand // TableForm
Print["primes under 50"]
poly[p_] := (x - 1)^p - (x^p - 1) // Expand;
coefflist[p_Integer] := Coefficient[poly[p], x, #] & /@ Range[0, p - 1];
AKSPrimeQ[p_Integer] := (Mod[coefflist[p] , p] // Union) == {0};
Select[Range[1, 50], AKSPrimeQ]
