reverseDigits[n_Integer] := FromDigits@Reverse@IntegerDigits@n


(* ========== *)

emirpQ[n_Integer] :=
 Block[{rev = reverseDigits@n}, And[n != rev, PrimeQ[rev]]]


(* ========== *)

nextEmirp[n_Integer] :=
 NestWhile[NextPrime, NextPrime[n], ! emirpQ[#] &]


(* ========== *)

Rest@NestList[nextEmirp, 1, 20]


(* ========== *)

Rest@NestWhileList[nextEmirp, 7700, # < 8000 &]


(* ========== *)

Nest[nextEmirp, 1, 10000]
