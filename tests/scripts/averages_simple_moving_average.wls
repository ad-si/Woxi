MA[x_List, r_] := Join[Table[Mean[x[[1;;y]]],{y,r-1}], MovingAverage[x,r]]


(* ========== *)

MAData = {{}, 0};
MAS[x_, t_: Null] :=
 With[{r = If[t === Null, MAData[[2]], t]},
  Mean[MAData[[1]] =
    If[Length[#] > (MAData[[2]] = r), #[[-r ;; -1]], #] &@
     Append[MAData[[1]], x]]]
