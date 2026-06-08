ClearAll[Combination, Permutation]
Combination[n_, k_] := Binomial[n, k]
Permutation[n_, k_] := Binomial[n, k]*k!

Print["Permutations P(n, 2):"]
Do[Print[n, " P 2 = ", Permutation[n, 2]], {n, 2, 6}]
Print["Combinations C(n, 2):"]
Do[Print[n, " C 2 = ", Combination[n, 2]], {n, 2, 6}]
