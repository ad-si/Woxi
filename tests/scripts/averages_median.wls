Median[{1, 5, 3, 2, 4}]
Median[{1, 5, 3, 6, 4, 2}]


(* ========== *)

mymedian[x_List]:=Module[{t=Sort[x],L=Length[x]},
 If[Mod[L,2]==0,
  (t[[L/2]]+t[[L/2+1]])/2
 ,
  t[[(L+1)/2]]
 ]
]


(* ========== *)

mymedian[{1, 5, 3, 2, 4}]
mymedian[{1, 5, 3, 6, 4, 2}]
