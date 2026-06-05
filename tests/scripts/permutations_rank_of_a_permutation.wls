(* Map between a permutation of {0,1,2,3} and its rank (Myrvold & Ruskey).
   `RankedMin[list, k]` is the k-th smallest element, i.e. `Sort[list][[k]]`. *)
fromrank[list_, 0] := list;
fromrank[list_, n_] :=
  Prepend[fromrank[DeleteCases[list, #], Mod[n, Factorial[Length[list] - 1]]], #] &@
   Sort[list][[Quotient[n, Factorial[Length[list] - 1]] + 1]];
rank[{}] = 0;
rank[{x_, y___}] := Count[{y}, _?(# < x &)] Factorial[Length[{y}]] + rank[{y}];
Print /@ Table[{n, fromrank[{0, 1, 2, 3}, n], rank@fromrank[{0, 1, 2, 3}, n]},
   {n, 0, 23}];
(* The task then prints a few random permutations of larger sets via
   `RandomInteger`; omitted here so the snapshot is deterministic. *)
