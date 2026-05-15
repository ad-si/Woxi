popcount[n_Integer] := IntegerDigits[n, 2] // Total
perniciousQ[n_Integer] := popcount[n] // PrimeQ
perniciouscount = 0;
perniciouslist = {};
i = 0;
While[perniciouscount < 25,
 If[perniciousQ[i], AppendTo[perniciouslist, i]; perniciouscount++];
 i++]
Print["first 25 pernicious numbers"]
perniciouslist
(*******)
perniciouslist2 = {};
Do[
 If[perniciousQ[i], AppendTo[perniciouslist2, i]]
 , {i, 888888877, 888888888}]
Print["Pernicious numbers between 888,888,877 and 888,888,888 (inclusive)"]
perniciouslist2


(* ========== *)

perniciousQ[n_Integer] := PrimeQ@Total@IntegerDigits[n, 2]


(* ========== *)

n = 0; NestWhile[Flatten@{#, If[perniciousQ[++n], n, {}]} &, {}, Length@# < 25 &]


(* ========== *)

Cases[Range[888888877, 888888888], _?(perniciousQ@# &)]
