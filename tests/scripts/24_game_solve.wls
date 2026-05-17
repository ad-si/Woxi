treeR[n_] := Table[o[trees[a], trees[n - a]], {a, 1, n - 1}]
treeR[1] := n
tree[n_] :=
 Flatten[treeR[n] //. {o[a_List, b_] :> (o[#, b] & /@ a),
    o[a_, b_List] :> (o[a, #] & /@ b)}]
game24play[val_List] :=
 Union[StringReplace[StringTake[ToString[#, InputForm], {10, -2}],
     "-1*" ~~ n_ :> "-" <> n] & /@ (HoldForm /@
      Select[Union@
        Flatten[Outer[# /. {o[q_Integer] :> #2[[q]],
             n[q_] :> #3[[q]]} &,
          Block[{O = 1, N = 1}, # /. {o :> o[O++], n :> n[N++]}] & /@
           tree[4], Tuples[{Plus, Subtract, Times, Divide}, 3],
          Permutations[Array[v, 4]], 1]],
       Quiet[(# /. v[q_] :> val[[q]]) == 24] &] /.
     Table[v[q] -> val[[q]], {q, 4}])]


(* ========== *)

game24play[RandomInteger[{1, 9}, 4]]


(* ========== *)

evaluate[HoldForm[op_[l_, r_]]] := op[evaluate[l], evaluate[r]];
evaluate[x_] := x;
combine[l_, r_ /; evaluate[r] != 0] := {HoldForm[Plus[l, r]],
   HoldForm[Subtract[l, r]], HoldForm[Times[l, r]],
   HoldForm[Divide[l, r]] };
combine[l_, r_] := {HoldForm[Plus[l, r]], HoldForm[Subtract[l, r]],
   HoldForm[Times[l, r]]};
split[items_] :=
  Table[{items[[1 ;; i]], items[[i + 1 ;; Length[items]]]}, {i, 1,
    Length[items] - 1}];
expressions[{x_}] := {x};
expressions[items_] :=
  Flatten[Table[
    Flatten[Table[
      combine[l, r], {l, expressions[sp[[1]]]}, {r,
       expressions[sp[[2]]]}], 2], {sp, split[items]}]];

(* Must use all atoms in given order. *)
solveMaintainOrder[goal_, items_] :=
  Select[expressions[items], (evaluate[#] == goal) &];
(* Must use all atoms, but can permute them. *)
solveCanPermute[goal_, items_] :=
  Flatten[Table[
    solveMaintainOrder[goal, pitems], {pitems,
     Permutations[items]}]];
(* Can use any subset of atoms. *)
solveSubsets[goal_, items_] :=
  Flatten[Table[
    solveCanPermute[goal, is], {is,
     Subsets[items, {1, Length[items]}]}], 2];

(* Demonstration to find all the ways to create 1/5 from {2, 3, 4, 5}. *)
solveMaintainOrder[1/5, Range[2, 5]]
solveCanPermute[1/5, Range[2, 5]]
solveSubsets[1/5, Range[2, 5]]
