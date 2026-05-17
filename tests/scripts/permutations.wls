(***Standard list functions:*)
fold[f_, x_, {}] := x
fold[f_, x_, {h_, t___}] := fold[f, f[x, h], {t}]
insert[L_, x_, n_] := Join[L[[;; n - 1]], {x}, L[[n ;;]]]

(***Generate all permutations of a list S:*)

permutations[S_] :=
 fold[Join @@ (Function[{L},
       Table[insert[L, #2, k + 1], {k, 0, Length[L]}]] /@ #1) &, {{}},
   S]


(* ========== *)

Permutations[{1,2,3,4}]
