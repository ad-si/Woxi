Print[Fibonacci[9]]

(******************************************************************************)

fib0[0] = 0
fib0[1] = 1
fib0[n_Integer] := fib0[n - 1] + fib0[n - 2]

Print[fib0[10]]

(******************************************************************************)

fib1[0] = 0
fib1[1] = 1
fib1[n_Integer] := fib1[n] = fib1[n - 1] + fib1[n - 2]

Print[fib1[11]]

(******************************************************************************)

fibi[prvprv_Integer, prv_Integer, rm_Integer] :=
  If[rm < 1, prvprv, fibi[prv, prvprv + prv, rm - 1]]
fib2[n_Integer] := fibi[0, 1, n]

Print[fib2[12]]

(******************************************************************************)

fib3[n_Integer] := Block[{tmp, prvprv = 0, prv = 1},
  For[i = 0, i < n, i++, tmp = prv; prv += prvprv; prvprv = tmp];
  Return[prvprv]]

Print[fib3[13]]

(******************************************************************************)

fibi[{prvprv_Integer, prv_Integer}] := {prv, prvprv + prv}
fibList[n_Integer] := Map[Take[#, 1] &, NestList[fibi, {0, 1}, n]] // Flatten

Print[fibList[20]]
